'''
train.py
'''

import code
import utils
import random

import torch
from torch import optim
from torch import nn

from networks import MachineTranslator


def train(mdl, exs, learning_rate=0.01, print_every=5000):
    optimizer = optim.SGD(list(mdl.encoder.parameters()) +
                          list(mdl.decoder.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()
    loss_total = 0

    print(f'================== Epoch {epoch} ==================')
    random.shuffle(exs)
    for ei, pair in enumerate(exs):
        optimizer.zero_grad()
        loss = mdl.forward(pair[0], output=pair[1], criterion=criterion)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() / pair[1].size(0)
        if ei % print_every == 0 and ei != 0:
            print('iter: {} ({:.2f}%) loss_avg: {:.4f}'
                    ''.format(ei, ei/len(exs)*100, loss_total/print_every))
            loss_total = 0

        if ei % 50000 == 0 and ei != 0:
            # Save
            torch.save({
                'enc.state_dict': model.encoder.state_dict(),
                'dec.state_dict': model.decoder.state_dict(),
                'in_lm': in_lm,
                'out_lm': out_lm,
                'max_length': MAX_LENGTH
            }, 'data/mt.mdl')


# main function
if __name__ == '__main__':
    # Set the hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LENGTH = 30
    n_epochs = 5

    # Prepare data
    in_lm, out_lm, pairs = utils.prepare_data('eng', 'spa', MAX_LENGTH, device)

    # Initiate the model
    model = MachineTranslator(in_lm, out_lm, device, max_length=MAX_LENGTH)

    # Train
    for epoch in range(n_epochs):
        train(model, pairs)
        code.interact(local=dict(locals(), **globals()))
        # Save
        torch.save({
            'enc.state_dict': model.encoder.state_dict(),
            'dec.state_dict': model.decoder.state_dict(),
            'in_lm': in_lm,
            'out_lm': out_lm,
            'max_length': MAX_LENGTH
        }, 'data/mt.mdl')

