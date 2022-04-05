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

with torch.no_grad():
    code.interact(local=dict(locals(), **globals()))
