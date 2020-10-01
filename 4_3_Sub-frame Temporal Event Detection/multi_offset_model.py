import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2

oneFloating=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

multi_factor = 2

class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True, step=1):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.step = step

        net = MobileNetV2(width_mult=width_mult)
        state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar', map_location=lambda storage, location: storage)
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            if not oneFloating:
                self.lin = nn.Linear(2*self.lstm_hidden, multi_factor*9)
            else:
                self.lin = nn.Linear(2*self.lstm_hidden, 2*8)
        else:
            if not oneFloating:
                self.lin = nn.Linear(self.lstm_hidden, multi_factor*9)
            else:
                self.lin = nn.Linear(self.lstm_hidden, 2*8)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(device), requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)

        # Transform logits to probabilities
        out = torch.sigmoid(out).clone()

        if not oneFloating:
            out = torch.reshape(out, [out.size()[0], out.size()[1], multi_factor, int(out.size()[2]/multi_factor)])
            # Add offset of each bin
            if True:
                out[:, :, :, 0] -= 0.5
                out[:, :, :, 0] *= self.step
                out[:, :, :, 0] += self.step * \
                                    torch.torch.arange(0,timesteps).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,
                                                                                                   multi_factor).to(device)
            else: # no offset
                out[:, :,:, 0] = self.step * \
                                 torch.torch.arange(0, timesteps).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,
                                                                                                   multi_factor).to(device)

            return torch.reshape(out, [out.size()[0], -1, out.size()[3]])
        else:
            out = torch.reshape(out, [out.size()[0], out.size()[1], -1, 2])

            # Add offset of each bin
            if True:
                out[:, :, :, 0] *= self.step
                out[:, :, :, 0] += self.step * \
                                    torch.torch.arange(0,timesteps).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,
                                                                                                   8).to(device)
            else: # no offset
                out[:, :,:, 0] = self.step * \
                                 torch.torch.arange(0, timesteps).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,
                                                                                                   8).to(device)

            final_out= torch.cat((out[:, :,:, 0:1],
                       torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(out.size()[0],out.size()[1],1,1).to(device) * out[:, :,:, 1:].repeat(1,1,1,8)) ,3)

            return torch.reshape(final_out, [final_out.size()[0], -1, final_out.size()[3]])






