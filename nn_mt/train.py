'''
train.py

A startup script. Actual training is launched by running this script.
Typically, this files handles setting up hyperparameters, runtime variables and handles all the high-level training
processes, such as loading datasets, running training loop, saving the trained model, and visualizing results.

'''

# importing system-level packages
import code
import utils
import random

# importing 3rd-party libraries
import torch
from torch import optim
from torch import nn

# importing local packages
from networks import MachineTranslator


def train(mdl, exs, learning_rate=0.01, print_every=5000):
    # Required components: 1. examples, 2. optimizer, 3. loss function, 4. (optional) scheduler
    optimizer = optim.SGD(list(mdl.encoder.parameters()) +
                          list(mdl.decoder.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss()
    loss_total = 0

    print(f'================== Epoch {epoch} ==================')
    random.shuffle(exs)
    for ei, pair in enumerate(exs):
        # The minimal requirements for one training step: 1. forward 2. backword, 3 optimizer_step 4. optimizer_zero
        loss = mdl.forward(pair[0], output=pair[1], criterion=criterion)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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


# Main function: This will run only when you execute this file in the command line such as `python train.py`
if __name__ == '__main__':
    # Set the hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine whether to use GPU or not
    MAX_LENGTH = 30  # Max length of an input sequence in words
    n_epochs = 5  # No. of epochs (One epoch is when an entire dataset is passed forward and backward in a training process)

    # Prepare data: Do preprocessing and return only what is needed for training.
    in_lm, out_lm, pairs = utils.prepare_data('eng', 'spa', MAX_LENGTH, device)

    # Initialized a model
    model = MachineTranslator(in_lm, out_lm, device, max_length=MAX_LENGTH)

    # Train
    for epoch in range(n_epochs):
        train(model, pairs)

        # Save a trained model per each epoch. We will save the model states in dictionaries, in/out language models,
        # and a hyperparameter
        torch.save({
            'enc.state_dict': model.encoder.state_dict(),
            'dec.state_dict': model.decoder.state_dict(),
            'in_lm': in_lm,
            'out_lm': out_lm,
            'max_length': MAX_LENGTH
        }, 'data/mt.mdl')

        # Note. What is state_dict?
        # In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in
        # the model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object
        # that maps each layer to its parameter tensor.

        # Note. Different use-cases of saving models.
        #
        # Case 1: save the model to use it yourself for inference
        #   torch.save(model.state_dict(), filepath)
        #   model.load_state_dict(torch.load(filepath))
        #   model.eval()
        #
        # Case 2: save model to resume training later. You need to save more than just a model, such as the state of
        # the optimizer, epochs, score, etc.
        #   state = {
        #       'epoch': epoch,
        #       'state_dict': model.state_dict(),
        #       'optimizer': optimizer.state_dict(),
        #       ...
        #   }
        #   torch.save(state, filepath)
        #
        # Case 3: Model to be used by someone else with no access to your code. You need to save both the model
        # architecture and parameters
        #   torch.save(model, filepath)
        #   model = torch.load(filepath)

