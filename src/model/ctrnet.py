import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from model.criterion import Criterion
from model.fpn_resnet import resnet50
from model.pred_decoder import pred_decoder


def basic_clf(pred_plg_info):
    """
    Input shape: (B, 11) float
    Output shape: (B,) bool
    """
    cond_1 = pred_plg_info[:, 0] > 0.6
    cond_2 = pred_plg_info[:, 1] > 5.
    cond_3 = pred_plg_info[:, 2] > 15.
    cond_4 = pred_plg_info[:, 3] > 0.5
    return cond_1 & cond_2 & cond_3 & cond_4


class CTRNet:

    def __init__(self, lr_initial, lr_scheduler_settings, class_weights, device_ids):
        # GPU settings.
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(i) for i in device_ids])

        # Model settings.
        self.fpn = nn.DataParallel(resnet50(pretrained=True, num_classes=6)).cuda()
        self.basic_clf = basic_clf
        self.svc, self.svc_func = None, None

        # Optimization settings.
        self.criterion = Criterion(class_weights)
        self.optimizer = torch.optim.Adam(self.fpn.parameters(), lr=lr_initial, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, *lr_scheduler_settings)

        self.current_epoch = 0
        self.best_f_prf = (-1., -1., -1.)
        self.best_svc_score = -1.
        self.prf_history = []
        self.img_tensor_all_train, self.y_plgs_all_train, self.ignored_plgs_all_train = None, None, None
        self.img_tensor_all_test, self.y_plgs_all_test, self.ignored_plgs_all_test = None, None, None
        self.tfps_all_train, self.pred_plgs_info_all_train = None, None
        self.tfps_all_test, self.pred_plgs_info_all_test = None, None

    def preload_img_plg(self, test_loader, test_mode=False):
        # Basic settings
        step_num = len(test_loader)
        img_tensor_all, y_plgs_all, ignored_plgs_all = [], [], []

        print('Loading data')
        for step, (img_tensor, y_plgs, ignored_plgs) in enumerate(test_loader):
            img_tensor_all.append(img_tensor)
            y_plgs_all.append(y_plgs)
            ignored_plgs_all.append(ignored_plgs)
            print('Current progress:', str(step + 1) + '/' + str(step_num), '<<<<<', end='\r')
        print('')

        if test_mode:
            self.img_tensor_all_test = img_tensor_all
            self.y_plgs_all_test = y_plgs_all
            self.ignored_plgs_all_test = ignored_plgs_all
        else:
            self.img_tensor_all_train = img_tensor_all
            self.y_plgs_all_train = y_plgs_all
            self.ignored_plgs_all_train = ignored_plgs_all

    def train_fpn(self, train_loader):
        """Train the neural network for an epoch"""
        # Basic settings
        loss_avg = 0.
        step_num = len(train_loader)

        # Train for an epoch
        print('Start training fpn')
        self.fpn.train()
        for step, (img_tensor, label_tensor, ignored_mask) in enumerate(train_loader):
            img_tensor = img_tensor.cuda()
            label_tensor = label_tensor.cuda()
            ignored_mask = ignored_mask.cuda()

            # Forward
            pred = self.fpn(img_tensor)
            loss = self.criterion(pred, label_tensor, ignored_mask)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Report
            loss_value = float(loss)
            loss_avg = (step * loss_avg + loss_value) / (step + 1.)
            print('Epoch:', self.current_epoch,
                  'Step:', str(step + 1) + '/' + str(step_num),
                  'Current loss: %.5f' % loss_value, '<<<<<', end='\r')
        print('')
        print('Best(F)_p: %.5f' % self.best_f_prf[0],
              'Best(F)_r: %.5f' % self.best_f_prf[1],
              'Best(F)_f: %.5f' % self.best_f_prf[2],
              'Average loss: %.5f' % loss_avg)
        self.lr_scheduler.step()
        self.current_epoch += 1
        return loss_avg

    def predict_and_record(self, decoder_settings, test_mode=False):
        img_tensor_all = self.img_tensor_all_test if test_mode else self.img_tensor_all_train
        y_plgs_all = self.y_plgs_all_test if test_mode else self.y_plgs_all_train
        ignored_plgs_all = self.ignored_plgs_all_test if test_mode else self.ignored_plgs_all_train

        step_num = len(img_tensor_all)
        tfps_all, pred_plgs_info_all = [], []
        total_time = 0.

        # Perform calculation
        print('Start calculation')
        self.fpn.eval()
        with torch.no_grad():
            for step in range(step_num):
                img_tensor = img_tensor_all[step].cuda()
                y_plgs, ignored_plgs = y_plgs_all[step], ignored_plgs_all[step]

                torch.cuda.synchronize()
                t_start = time.time()

                # Forward
                pred = self.fpn(img_tensor)
                pred[:, 0:3] = torch.sigmoid(pred[:, 0:3])

                # Decode
                pred_plgs, pred_plgs_info = pred_decoder(pred.cpu().numpy()[0], decoder_settings)

                torch.cuda.synchronize()
                total_time += time.time() - t_start

                # Get tfps
                tfps = np.zeros((len(pred_plgs),), dtype='float64')
                for idx, pred_plg in enumerate(pred_plgs):
                    for ignored_plg in ignored_plgs:
                        if pred_plg.iou_with(ignored_plg) > 0.5:
                            tfps[idx] = -1.
                            break
                    else:
                        for y_plg in y_plgs:
                            if pred_plg.iou_with(y_plg) > 0.5:
                                tfps[idx] = 1.
                                break
                        else:
                            tfps[idx] = 0.

                # Filter out ignored tfps and pred_plgs while appending them.
                tfps_all.append(tfps[tfps != -1.])
                pred_plgs_info_all.append(pred_plgs_info[tfps != -1.])
                print('Current progress:', str(step + 1) + '/' + str(step_num), '<<<<<', end='\r')
        self.fpn.train()
        print('')
        print('FPS: %.5f' % (step_num / total_time))

        if test_mode:
            self.tfps_all_test, self.pred_plgs_info_all_test = tfps_all, pred_plgs_info_all
        else:
            self.tfps_all_train, self.pred_plgs_info_all_train = tfps_all, pred_plgs_info_all

    def train_svc(self):
        x, y = map(np.concatenate, (self.pred_plgs_info_all_train, self.tfps_all_train), (0, 0))

        mask_basic = self.basic_clf(x)
        x, y = x[mask_basic], y[mask_basic]

        svc = GridSearchCV(estimator=SVC(kernel='rbf'),
                           param_grid={'class_weight': ({0: 2., 1: 1.}, {0: 1.5, 1: 1.}, {0: 1., 1: 1.},
                                                        {0: 1., 1: 1.5}, {0: 1., 1: 2.}),
                                       'C': (0.5, 1., 10., 100.),
                                       'gamma': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)},
                           cv=5, n_jobs=32, verbose=1)
        svc.fit(x[:, [0, 3, 4]], y)

        def svc_func(plg_info):
            if len(plg_info) == 0:
                return np.array([], dtype='bool')
            return svc.predict(plg_info[:, [0, 3, 4]]).astype('bool')

        if self.svc is None:
            self.svc = svc
            self.svc_func = svc_func
        else:
            svc_previous = self.svc
            svc_func_previous = self.svc_func
            f_previous = self.report_prf()[2]
            self.svc = svc
            self.svc_func = svc_func
            if f_previous > self.report_prf()[2]:
                self.svc = svc_previous
                self.svc_func = svc_func_previous

    def report_prf(self):
        p, r = 0., 0.
        step_num = len(self.tfps_all_test)
        for idx in range(step_num):
            y_plgs = self.y_plgs_all_test[idx]
            tfps = self.tfps_all_test[idx]
            pred_plgs_info = self.pred_plgs_info_all_test[idx]

            tfps = tfps[self.basic_clf(pred_plgs_info) & self.svc_func(pred_plgs_info)]
            tp, tp_fp, tp_fn = sum(tfps), len(tfps), len(y_plgs)

            # Calculate precision and recall
            _p, _r = (1., 1.) if (tp_fp == 0) and (tp_fn == 0) else \
                     (1., 0.) if (tp_fp == 0) and (tp_fn != 0) else \
                     (0., 1.) if (tp_fp != 0) and (tp_fn == 0) else \
                     (tp / tp_fp, tp / tp_fn)
            p += _p
            r += _r
        p /= step_num
        r /= step_num
        f = 0. if (p + r) == 0. else 2. * p * r / (p + r)
        print('Current PRF: (%.5f, %.5f, %.5f)' % (p, r, f))
        return p, r, f

    def compare_and_save(self):
        p, r, f = self.report_prf()
        self.prf_history.append((self.current_epoch - 1, (p, r, f)))
        if f > self.best_f_prf[2]:
            print('F score has improved from: (%.5f, %.5f, %.5f)' % self.best_f_prf)
            self.best_f_prf = (p, r, f)
            self.save_weights('sn_best_f.weights')
            self.save_svc('sn_best_f.svc')
        else:
            print('F score has not improved from: (%.5f, %.5f, %.5f)' % self.best_f_prf)

    def save_weights(self, f_name):
        path = './../saved_weights/' + f_name
        print('Saving weights to', path, end='...')
        torch.save(self.fpn.module.state_dict(), path)
        print('Succeeded')

    def load_weights(self, f_name):
        path = './../saved_weights/' + f_name
        print('Loading weights from', path, end='...')
        self.fpn.module.load_state_dict(torch.load(path, map_location='cuda:0'))
        print('Succeeded')

    def save_svc(self, f_name):
        path = os.path.join('./../saved_svc', f_name)
        print('Saving svc to', path, end='...')
        with open(path, 'wb') as f:
            pickle.dump(self.svc, f)
        print('Succeeded')

    def load_svc(self, f_name):
        path = os.path.join('./../saved_svc', f_name)
        print('Loading svc from', path, end='...')
        with open(path, 'rb') as f:
            svc = pickle.load(f)

        def svc_func(plg_info):
            if len(plg_info) == 0:
                return np.array([], dtype='bool')
            return svc.predict(plg_info[:, [0, 3, 4]]).astype('bool')

        self.svc, self.svc_func = svc, svc_func
        print('Succeeded')
