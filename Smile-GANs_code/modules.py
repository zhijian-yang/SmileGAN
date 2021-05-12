import torch
import torch.nn as nn
import numpy as np

class TwoInputModule(nn.Module):
    def forward(self, input1, input2):
        raise NotImplementedError


class TwoInputSequential(nn.Sequential, TwoInputModule):
    def __init__(self, *args):
        super(TwoInputSequential, self).__init__(*args)

    def forward(self, input1, input2):
        ## overloads forward function in parent class
        for module in self._modules.values():
            if isinstance(module, TwoInputModule): #check whether it is twoinputmodule
                input1 = module.forward(input1, input2)
            else:
                input1 = module.forward(input1)
        return input1


class Sub_Adder(TwoInputModule):
    def __init__(self, x_dim, z_dim):
        super(Sub_Adder, self).__init__()
        self.add_noise = nn.Sequential(
            nn.Linear(z_dim, x_dim),
        )

    def forward(self, input, noise):
        multiplier = torch.sigmoid(self.add_noise.forward(noise))
        return input*multiplier





