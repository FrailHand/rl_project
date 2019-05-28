#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):

    def __init__(self, h, w, c, outputs):
        super(DRQN, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, self.conv1.kernel_size[0], self.conv1.stride[0]),
                                self.conv2.kernel_size[0], self.conv2.stride[0])
        convh = conv2d_size_out(conv2d_size_out(h, self.conv1.kernel_size[1], self.conv1.stride[1]),
                                self.conv2.kernel_size[1], self.conv2.stride[1])

        linear_input_size = convw * convh * self.conv2.out_channels
        self.head = nn.LSTM(linear_input_size, outputs, batch_first=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, hidden_in=None):
        batch = x.shape[0]
        time = x.shape[1]

        if x.dim() < 5:
            time = 1

        x = x.view(batch * time, self.c, self.h, self.w)
        x = F.relu(self.bn1(self.conv1(x)))  # Add a batch dimension.
        x = F.relu(self.bn2(self.conv2(x)))
        output, hidden = self.head(x.view(batch, time, -1), hidden_in)
        output = torch.squeeze(output)

        return output, hidden
