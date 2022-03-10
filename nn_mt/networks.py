'''
networks.py
'''

import code
import random
import torch
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, device, hidden_size=256):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.to(device)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros((1, 1, self.hidden_size), device=self.device)
    
class DecoderRNN(nn.Module):
    """Attention Decoder"""
    def __init__(self, output_size, device, max_length, hidden_size=256, dropout_p=0.05):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size *2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.to(device)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Decoder Attention layer
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Attention over encoder outputs 
        x_ = torch.cat((embedded[0], attn_applied[0]), 1)
        x_ = self.attn_combine(x_).unsqueeze(0)
        x_ = F.relu(x_)

        # RNN
        output, hidden = self.gru(x_, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights


class MachineTranslator:
    def __init__(self, in_lm, out_lm, device, max_length=30, 
                 teacher_forcing_ratio=0.3):
        self.device = device
        self.in_lm = in_lm
        self.out_lm = out_lm
        self.max_length = max_length
        self.tf_ratio = teacher_forcing_ratio
        self.encoder_input_size = len(in_lm.word2index)
        self.decoder_input_size = len(out_lm.word2index)
        self.encoder = EncoderRNN(self.encoder_input_size, device)
        self.decoder = DecoderRNN(self.decoder_input_size, device, max_length)
    
    def forward(self, input, output=None, criterion=None):
        loss = 0
        enc_outputs = torch.zeros(self.max_length, self.encoder.hidden_size,
                                  device=self.device)
        enc_hidden = self.encoder.initHidden()

        for ei in range(input.size(0)):
            enc_output, enc_hidden = self.encoder(input[ei], enc_hidden)
            enc_outputs[ei] = enc_output[0, 0]

        # Decoder
        dec_input = torch.tensor([[0]], device=self.device)
        dec_hidden = enc_hidden

        output_len = output.size(0) if output is not None else self.max_length
        output_str = []
        if output is not None and random.random() < self.tf_ratio:
            # Teacher forcing: Feed the target as the next input
            for di in range(output_len):
                decoder_output, dec_hidden, decoder_attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)
                loss += criterion(decoder_output, output[di])
                dec_input = output[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(output_len):
                decoder_output, dec_hidden, decoder_attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)
                topv, topi = decoder_output.topk(1)
                output_str.append(topi)
                dec_input = topi.squeeze().detach()  # detach from history as input

                if criterion is not None:
                    loss += criterion(decoder_output, output[di])
                if dec_input.item() == 1:  # 1 for eos
                    break

        if output is None:
            return loss, [self.out_lm.index2word[wi] for wi in output_str]
        return loss
