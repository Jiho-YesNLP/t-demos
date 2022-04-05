'''
test.py
'''
import code

import torch
from torch import optim
from torch import nn

from networks import MachineTranslator

import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved = torch.load('data/mt1.mdl')
model = MachineTranslator(saved['in_lm'], saved['out_lm'], device, 
                          max_length=saved['max_length'])
model.encoder.load_state_dict(saved['enc.state_dict'])
model.decoder.load_state_dict(saved['dec.state_dict'])

def translate(input):
    s = utils.normalizeString(input[:saved['max_length']])
    s = saved['in_lm'].to_tensor(s).to(device)
    translated = model.forward(s)
    print('translated: {}'.format(' '.join(translated[1])))

# Comments by PyTorch.org albanD
model.eval()  # eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will
              # work in eval model instead of training mode
with torch.no_grad():  # no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed
                       # up computations but you won't be able to backprop
    code.interact(local=dict(locals(), **globals()))
