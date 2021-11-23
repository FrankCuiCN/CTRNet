from loaders.test_loader import TestLoader
from loaders.train_loader import TrainLoader
from model.ctrnet import CTRNet

# Data settings.
dataset = 'ctw1500'
short_side = 640.

# Training settings.
batch_size = 32
class_weights = (10., 1., 1., 1., 1., 1.)
device_ids = (0, 1, 2, 3)
lr_initial = 0.0001
lr_scheduler_settings = ([100, 200], 1)

# Evaluation settings.
decoder_settings = (0.65, 0.)
eval_start_point, eval_interval = 10, 5

# Data prep.
dataset_train, dataset_test = dataset + '_train', dataset + '_test'
train_loader = TrainLoader(dataset_train, batch_size, short_side=short_side, num_workers=32)
test_loader_train = TestLoader(dataset_train, short_side=short_side, num_workers=32)
test_loader_test = TestLoader(dataset_test, short_side=short_side, num_workers=32)

# Model def.
ctrnet = CTRNet(lr_initial, lr_scheduler_settings, class_weights, device_ids)
ctrnet.preload_img_plg(test_loader_train, test_mode=False)
ctrnet.preload_img_plg(test_loader_test, test_mode=True)

# Training.
for epoch in range(6000):
    ctrnet.train_fpn(train_loader)
    if (ctrnet.current_epoch >= eval_start_point) and (ctrnet.current_epoch % eval_interval) == 0:
        ctrnet.predict_and_record(decoder_settings, test_mode=True)
        # Train svc.
        ctrnet.predict_and_record(decoder_settings, test_mode=False)
        ctrnet.train_svc()
        # Report, compare, and save.
        ctrnet.compare_and_save()
