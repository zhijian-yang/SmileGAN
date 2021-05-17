import os
import time
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from .model import SmileGAN
from .evaluate import eval_w_distances, cluster_output, label_change
from .utils import Covariate_correction, Data_normalization, parse_train_data, parse_validation_data

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Smile_GAN_train():
    
    def __init__(self, ncluster, start_saving_epoch, max_epoch, WD_threshold, AQ_threshold, \
        cluster_loss_threshold, lam=9, mu=5, batchSize=25, lipschitz_k = 0.5, \
        beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 25, save_epoch_freq = 5, print_freq = 1000):
        self.opt=dotdict({})
        self.opt.ncluster = ncluster
        self.opt.start_saving_epoch = start_saving_epoch
        self.opt.max_epoch = max_epoch
        self.opt.WD_threshold = WD_threshold
        self.opt.AQ_threshold = AQ_threshold
        self.opt.cluster_loss_threshold = cluster_loss_threshold
        self.opt.lam = lam
        self.opt.mu = mu
        self.opt.batchsize = batchSize
        self.opt.lipschitz_k = lipschitz_k
        self.opt.beta1 = beta1
        self.opt.lr = lr
        self.opt.max_gnorm = max_gnorm
        self.opt.save_epoch_freq = save_epoch_freq
        self.opt.print_freq = print_freq
        self.opt.eval_freq = eval_freq

    def print_log(self, result_f, message):
        result_f.write(message+"\n")
        result_f.flush()
        print(message)

    def format_log(self, epoch, epoch_iter, measures, t, prefix=True):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
        if not prefix:
            message = ' ' * len(message)
        for key, value in measures.items():
            message += '%s: %.4f ' % (key, value)
        return message

    def parse_data(self, data, covariate, random_seed, data_fraction):
        cn_train_dataset, pt_train_dataset = parse_train_data(data, covariate, random_seed, data_fraction, self.opt.batchsize)
        cn_eval_dataset, pt_eval_dataset = parse_validation_data(data, covariate)
        self.opt.nROI = pt_eval_dataset.shape[1]
        self.opt.n_val_data = pt_eval_dataset.shape[0]
        return cn_train_dataset, pt_train_dataset, cn_eval_dataset, pt_eval_dataset


    def train(self, model_name, data, covariate, save_dir, random_seed=0, data_fraction=1, verbose=True, independent_ROI = True):
        if not verbose: result_f = open("%s/results.txt" % save_dir, 'w')

        cn_train_dataset, pt_train_dataset, eval_X, eval_Y = self.parse_data(data, covariate, random_seed, data_fraction)
  
        # create_model
        model = SmileGAN()
        model.create(self.opt)

        total_steps = 0
        print_start_time = time.time()
        max_distance_list = [0,0]                    ##### number of consecutive epochs with max_wd < threshold
        aq_loss_cluster_list = [[0 for _ in range(4)] for _ in range(2)]                    ##### number of consecutive epochs with aq and cluster_loss < threshold
        savetime=0                                      ##### keep track of number of times that stopping criteria is satisfied if choosing not to stop model
        predicted_label_past=np.zeros(self.opt.n_val_data)   ##### keep track of assigned clustering membership in epochs
        if not verbose:
            pbar = tqdm(total = self.opt.max_epoch)
        for epoch in range(1, self.opt.max_epoch + 1):
            if not verbose:
                pbar.update(1)
            epoch_start_time = time.time()
            epoch_iter = 0
            for i, pt_data in enumerate(pt_train_dataset):
                cn_data = cn_train_dataset.next()
                real_X, real_Y = Variable(cn_data['x']), Variable(pt_data['y'])

                total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize

                losses = model.train_instance(real_X, real_Y)

                if total_steps % self.opt.print_freq == 0:
                    t = (time.time() - print_start_time) / self.opt.batchsize
                    if verbose:
                        self.print_log(result_f, self.format_log(epoch, epoch_iter, losses, t))
                        print_start_time = time.time()

            if epoch % self.opt.save_epoch_freq == 0 and verbose:
                self.print_log(result_f, 'saving the model at the end of epoch %d.' % (epoch))
                model.save(save_dir,'latest')

            if epoch % self.opt.eval_freq == 0:
                t = time.time()
            
                predicted_label,predicted_class = label_change(model,eval_Y,self.opt)
                total_label_change = np.count_nonzero(np.absolute(np.array(predicted_label)-np.array(predicted_label_past))) 
                predicted_label_past = predicted_label
                max_distance, w_distances = eval_w_distances(eval_X, eval_Y, model, independent_ROI)
                max_distance_list.append(max_distance)
                max_distance_list.pop(0)
                aq_loss_cluster_list[0].append(total_label_change)
                aq_loss_cluster_list[1].append(losses['loss_cluster'])
                for _ in range(2):
                    aq_loss_cluster_list[_].pop(0) 

            
                t = time.time() - t
                res_str_list = ["[%d], Mean_W_Distance: %.4f, TIME: %.4f" % (epoch,max_distance, t)]
                res_str_list.extend(['W_Distances: %s' % [round(ele, 4) for ele in w_distances],'Subtype_Quantity: %s' %predicted_class])

                if len(w_distances)==model.opt.ncluster:
                    if max(max_distance_list) < self.opt.WD_threshold and max(aq_loss_cluster_list[0]) < self.opt.AQ_threshold\
                     and max(aq_loss_cluster_list[1])<self.opt.cluster_loss_threshold and epoch > self.opt.start_saving_epoch:
                        savetime+=1
                        model.save(save_dir, model_name)
                        pbar.close()
                        res_str_list += ["*** Stopping Criterion Satisfied ***"]
                        res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
                        self.print_log(result_f, res_str)
                        result_f.close()
                        return True

                res_str = "\n".join(["-"*60] + res_str_list + ["-"*60])
                res_str_list += ["*** Max iteration reached and criterion not satisfied ***"]
                if verbose:  
                    self.print_log(result_f, res_str)
        result_f.close()
        pbar.close()
        return False
