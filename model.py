import torch
import torch.nn as nn

from torchvision.models import DenseNet
from collections import OrderedDict

class Model(DenseNet):
    def __init__(self, cuda=True):
        super(Model, self).__init__()

        # Architecture
        # print(self.state_dict().keys())
        # Load pre-trained model
        self.load_weights('weights.pth', cuda=cuda)

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda and torch.cuda.is_available() else "cpu")

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in pretrained_model.items():
            if name not in own_state:
                na = name.replace("module.", "")
                new_state[na] = param
       
        pretrained_model = new_state
        # Load pre-trained weights in current model 
        with torch.no_grad():
            self.load_state_dict(new_state, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    # def forward(self, x):
    #     # TODO
    #     return self.forward(x)
    def forward(self, x):
        out = super(Model, self).forward(x)
        # print(out.shape)
        return out
