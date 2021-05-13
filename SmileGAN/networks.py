import torch.nn as nn
import torch.nn.functional as F
from .modules import TwoInputSequential, Sub_Adder

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)

def define_Linear_Mapping(nROI,nCluster):
    netG = LMappingGenerator(nCluster, nROI)
    netG.apply(weights_init)
    return netG

def define_Linear_Discriminator(nROI,nCluster):
    netD=LDiscriminator(nROI,nCluster)
    netD.apply(weights_init)
    return netD

def define_Linear_Clustering(nROI,nCluster):
    netC = LEncoder(nCluster,nROI)
    netC.apply(weights_init)
    return netC

##############################################################################
# Network Classes
##############################################################################

class LMappingGenerator(nn.Module):
    def __init__(self, nCluster, nROI, product_layer=Sub_Adder):
        super(LMappingGenerator, self).__init__()
        model=[]
        def block(in_layer, out_layer, normalize=False):
            layers = [nn.Linear(in_layer, out_layer,bias=False)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_layer, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        model+=block(nROI,int(nROI/2))+block(int(nROI/2), int(nROI/4))
        model.append(product_layer(int(nROI/4),nCluster))
        model+=block(int(nROI/4),int(nROI/2))+block(int(nROI/2),nROI)
        model.append(nn.Linear(nROI, nROI,bias=False))
        self.model = TwoInputSequential(*model)
    def forward(self, input_x,input_z):
        return self.model(input_x,input_z)

class LEncoder(nn.Module):
    def __init__(self, nCluster, nROI):
        super(LEncoder, self).__init__()
        model=[]
        def block(in_layer, out_layer, normalize=True):
            layers=[nn.LeakyReLU(0.2, inplace=True)]
            layers.append(nn.Linear(in_layer, out_layer))
            return layers
        model.append(nn.Linear(nROI, nROI))
        model+=block(nROI, int(nROI/2), normalize=False)+block(int(nROI/2), int(nROI/4))+block(int(nROI/4), nCluster)
        self.model = nn.Sequential(*model)

    def forward(self, input_y):
        z = self.model(input_y)
        z_softmax = F.softmax(z, dim=1)
        return z_softmax, z

class LDiscriminator(nn.Module):
    def __init__(self, nROI, nCluster):
        super(LDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nROI, int(nROI/2),bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(nROI/2), int(nROI/4),bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(nROI/4), 2,bias=True),
        )
    def forward(self, input_y):
        pred = self.model(input_y)
        return pred


