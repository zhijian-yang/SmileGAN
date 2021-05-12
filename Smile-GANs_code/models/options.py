import argparse
import os
import torch

class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        ##### directories
        self.parser.add_argument('--data_root', type=str, required=True, help='path to data')
        self.parser.add_argument('--save_dir', type=str, required=True, help='directory under which checkpoints, results and training procedures are saved')

        ##### data
        #self.parser.add_argument('--n_train_data', type=int, default=600,help='number of training datapoints')
        #self.parser.add_argument('--n_val_data', type=int, default=600,help='number of validation datapoints')

        ##### training
        #self.parser.add_argument('--batchSize', type=int, default=25, help='batch size')
        #self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        #self.parser.add_argument('--max_epoch', type=int, default=6000, help='maximum number of epochs')
        #self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        #self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for adam')
        #self.parser.add_argument('--lam', type=float, default=9, help='parameter for cluster loss')
        #self.parser.add_argument('--mu', type=float, default=5, help='parameter for change loss')

        ##### model
        #self.parser.add_argument('--ncluster', type=int, default=3, help='# of predefined subtypes')
        #self.parser.add_argument('--nROI', type=int, default=145, help='# of ROIS')
        #self.parser.add_argument('--max_gnorm', type=float, default=100, help='max grad norm to which it will be clipped (if exceeded)')

        ##### monitoring
        #self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        #self.parser.add_argument('--eval_freq', type=int, default=5, help='frequency of evaluating on validation set')
        #self.parser.add_argument('--load', type=bool, default=False, help='flag set to load previous checkpoint')
        #self.parser.add_argument('--view_synthetic_accuracy',type=bool, default=True, help='flag set to visualize clustering performance on synthetic data during training procedure')
        #self.parser.add_argument('--stop',type=bool, default=False, help='flag set to stop training when criteria met')
        #self.parser.add_argument('--independent_ROI',type=bool, default=True, help='flag set to assume independence of ROIs')
        #self.parser.add_argument('--WD_threshold',type=float, default=0.07, help='stopping threshold for maximum wasserstein distance')
        #self.parser.add_argument('--AQ_threshold',type=float, default=30, help='stopping threshold for alteration quantity')
        #self.parser.add_argument('--cluster_loss_threshold',type=float, default=0.0015, help='stopping threshold for cluter loss')
        
        self.initialized = True

    def parse(self, sub_dirs=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

