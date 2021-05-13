import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from itertools import chain as ichain
from .networks import define_Linear_Mapping, define_Linear_Clustering, define_Linear_Discriminator

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


#####sample from discrete uniform random variable and construct SUB variable. 
def sample_z(real_X, ncluster, fix_class=-1):
    """
    sample from discrete uniform random variable and construct SUB variable.

    :param real_X, torch tensor with real data 
    :param ncluster: int, defined number of clusters
    :param fix_class: int, can be set to certain mapping directions to generate data in only one cluster,
                           set to -1 for random sampling from discrete uniform distribution

    :return tensor with sampled or selected mapping directions & 
            tensors with shape n*k (each row is a one-hot vector depending on sampled directions) 
    """
    Tensor = torch.FloatTensor
    z = Tensor(real_X.size(0), ncluster).fill_(0)
    z_idx = torch.empty(real_X.size(0), dtype=torch.long)
    if (fix_class == -1):
        z_idx = z_idx.random_(ncluster)
        z = z.scatter_(1, z_idx.unsqueeze(1), 1.)
    else:
        z_idx[:] = fix_class
        z = z.scatter_(1, z_idx.unsqueeze(1), 1.)
        z[0,fix_class]
    z_var = Variable(z)
    return z_var, z_idx

def criterion_GAN(pred, target_is_real):
    if target_is_real:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(0.))
        loss=F.cross_entropy(pred, target_var)
    else:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(1.))
        loss = F.cross_entropy(pred, target_var)
    return loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class SmileGAN(object):
    def __init__(self):
        self.opt = None

        ##### definition of all netwotks
        self.netMapping = None
        self.netClustering= None
        self.netDiscriminator = None

        ##### definition of all optimizers
        self.optimizer_M = None
        self.optimizer_D = None

        ##### definition of all criterions
        self.criterionGAN = criterion_GAN
        self.criterionChange = F.l1_loss
        self.criterionCluster=F.cross_entropy


    def create(self, opt):

        self.opt = opt

        ## definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI,self.opt.ncluster)
        self.netClustering = define_Linear_Clustering(self.opt.nROI,self.opt.ncluster)
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI,self.opt.ncluster)

        ## definition of all optimizers
        self.optimizer_M = torch.optim.Adam(ichain(self.netMapping.parameters(),self.netClustering.parameters()),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
        self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
                                            lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))


    def train_instance(self, x, real_y):
        z_var, z_index = sample_z(x, self.opt.ncluster)
        fake_y = self.netMapping.forward(x,z_var)+x

        ## Discriminator loss
        pred_fake_y = self.netDiscriminator.forward(fake_y.detach())
        loss_D_fake_y = self.criterionGAN(pred_fake_y, False)
        pred_true_y = self.netDiscriminator.forward(real_y)
        loss_D_true_y = self.criterionGAN(pred_true_y, True)
        loss_D= 0.5* (loss_D_fake_y + loss_D_true_y)

        ## update weights of discriminator
        self.optimizer_D.zero_grad()
        loss_D.backward()
        gnorm_D = torch.nn.utils.clip_grad_norm_(self.netDiscriminator.parameters(), self.opt.max_gnorm)
        self.optimizer_D.step()

        ## Mapping/Change/Cluster loss
        pred_fake_y = self.netDiscriminator.forward(fake_y)
        loss_mapping = self.criterionGAN(pred_fake_y, True)

        reconst_z_softmax,reconst_z = self.netClustering.forward(fake_y)
        cluster_loss = self.criterionCluster(reconst_z, z_index)

        change_loss= self.criterionChange(fake_y, x)
        loss_G = loss_mapping+self.opt.lam*cluster_loss+self.opt.mu*change_loss

        ## update weights of mapping and clustering function
        self.optimizer_M.zero_grad()
        loss_G.backward()
        gnorm_M = torch.nn.utils.clip_grad_norm_(self.netMapping.parameters(), self.opt.max_gnorm)
        self.optimizer_M.step()

        ## perform weight clipping
        for p in self.netMapping.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)
        for p in self.netClustering.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)

        ## Return dicts
        losses=OrderedDict([('Discriminator_loss', loss_D.item()),('Mapping_loss', loss_mapping.item()),('loss_change', change_loss.item()),('loss_cluster', cluster_loss.item())])

        return losses
    
    ## return a numpy array of pattern type probabilities for all input subjects
    def predict_cluster(self,real_y):
        prediction,z=self.netClustering.forward(real_y)
        return prediction.detach().numpy()

    ## return generated patient data with given zub variable
    def predict_Y(self, x, z):
        return self.netMapping.forward(x, z)+x

    ## save checkpoint    
    def save(self, save_dir, chk_name):
        chk_path = os.path.join(save_dir, chk_name)
        checkpoint = {
            'netMapping':self.netMapping.state_dict(),
            'netDiscriminator':self.netDiscriminator.state_dict(),
            'optimizer_D':self.optimizer_D.state_dict(),
            'optimizer_M':self.optimizer_M.state_dict(),
            'netClustering':self.netClustering.state_dict(),
        }
        checkpoint.update(self.opt)
        torch.save(checkpoint, chk_path)

    def load_opt(self,checkpoint):
        self.opt = dotdict({})
        for key in checkpoint.keys():
            if key not in ['netMapping','netDiscriminator','netClustering','optimizer_M','optimizer_D']:
                self.opt[key] = checkpoint[key]
        

    ## load trained model
    def load(self, chk_path):
        checkpoint = torch.load(chk_path)
        self.load_opt(checkpoint)
        ##### definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI,self.opt.ncluster)
        self.netClustering = define_Linear_Clustering(self.opt.nROI,self.opt.ncluster)
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI,self.opt.ncluster)

        ##### definition of all optimizers
        self.optimizer_M = torch.optim.Adam(ichain(self.netMapping.parameters(),self.netClustering.parameters()),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
        self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
                                        lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))
        
        self.netMapping.load_state_dict(checkpoint['netMapping'])
        self.netDiscriminator.load_state_dict(checkpoint['netDiscriminator'])
        self.netClustering.load_state_dict(checkpoint['netClustering'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.optimizer_M.load_state_dict(checkpoint['optimizer_M'])
        self.load_opt(checkpoint)
            


        
