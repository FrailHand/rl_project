#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import torch.nn as nn
import torch.nn.functional as F


def normal_init(layers_):
    for layer in layers_.modules():
        classname = layer.__class__.__name__

        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            print('[INFO] (normal_init) classname {}'.format(classname))
            layer.weight.data.normal_(0.0, 0.004)
            layer.bias.data.fill_(0.0)


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        print('[INFO] ({}) Input h= {} w= {} c= {}'.format(self.__class__.__name__, h, w, c))

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w, self.conv1.kernel_size[0], self.conv1.stride[0]),
                                self.conv2.kernel_size[0], self.conv2.stride[0])
        convh = conv2d_size_out(conv2d_size_out(h, self.conv1.kernel_size[1], self.conv1.stride[1]),
                                self.conv2.kernel_size[1], self.conv2.stride[1])
        linear_input_size = convw * convh * self.conv2.out_channels
        # self.head = nn.Sequential(*[nn.Linear(linear_input_size, outputs), nn.Softmax()])
        self.head = nn.Linear(linear_input_size, outputs)

        normal_init(self.conv1)
        normal_init(self.conv2)
        normal_init(self.head)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, hidden_in=None):
        x = F.relu(self.bn1(self.conv1(x)))  # Add a batch dimension.
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.shape[0], -1)), None


class DQNLatent(nn.Module):
    def __init__(self, latent_dim, outputs):
        super(DQNLatent, self).__init__()

        self.input_dim = latent_dim
        self.output_dim = outputs
        self.f1_dim = 1024
        self.f2_dim = 1024

        print('[INFO] ({}) Input dimension {}'.format(self.__class__.__name__, self.input_dim))
        print('[INFO] ({}) Output dimension {}'.format(self.__class__.__name__, self.output_dim))

        self.back = nn.Sequential(*[nn.Linear(self.input_dim, self.f1_dim),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.f1_dim),
                                    nn.Linear(self.f1_dim, self.f2_dim),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.f2_dim),
                                    nn.Linear(self.f2_dim, self.output_dim)])
        # nn.Softmax()])

        normal_init(self.back)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, hidden=None):
        # x= x.view(self.input_dim, -1)
        # print('{} {}'.format(self.__class__.__name__, x.size()))
        return self.back(x), None
