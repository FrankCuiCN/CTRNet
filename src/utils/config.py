from utils.label_decoders import *

config_dict = {'ctw1500_train': ('./../datasets/ctw1500/data_train',
                                 './../datasets/ctw1500/labels_train',
                                 './../datasets/ctw1500/cache_train',
                                 get_pts_ctw1500),
               'ctw1500_test': ('./../datasets/ctw1500/data_test',
                                './../datasets/ctw1500/labels_test',
                                None,
                                get_pts_ctw1500)}
