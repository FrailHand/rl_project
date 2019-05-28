#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

from __future__ import print_function

import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def conv2d_size_out(size, kernel_size=5, stride=2, padding=0, dilation=1):
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class FFVAE(nn.Module):
    def __init__(self, dim_C, dim_H, dim_W, nb_hidden=20):
        super().__init__()

        self.dim_C = dim_C
        self.dim_H = dim_H
        self.dim_W = dim_W
        self.dim_flat = dim_C * dim_H * dim_W

        self.fc1 = nn.Linear(self.dim_flat, 400)
        self.fc21 = nn.Linear(400, nb_hidden)
        self.fc22 = nn.Linear(400, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 400)
        self.fc4 = nn.Linear(400, self.dim_flat)

    def encode(self, x):
        x = x.view(-1, self.dim_flat)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1, self.dim_C, self.dim_H, self.dim_W)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim_flat))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvVAE(nn.Module):
    def __init__(self, dim_C, dim_H, dim_W, z_size=20):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(dim_C, 32, 4, stride=2),
            nn.ReLU(),
            # nn.Conv2d(32, 64, 4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, 4, stride=2),
            # nn.ReLU(),
            nn.Conv2d(32, 256, 4, stride=2),
            nn.ReLU()
        )

        convh = conv2d_size_out(conv2d_size_out(dim_H, 4, 2), 4, 2)
        convw = conv2d_size_out(conv2d_size_out(dim_W, 4, 2), 4, 2)
        enc_out_dim = convh * convw * 256
        self.fc_mu = nn.Linear(enc_out_dim, z_size)
        self.fc_logvar = nn.Linear(enc_out_dim, z_size)

        self.fc_dec = nn.Linear(z_size, enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(enc_out_dim, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, 5, stride=2),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, 6, stride=2),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, dim_C, 4, stride=4),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        # print('[INFO] (encode) h size', h.size())
        return self.fc_mu(h.view(h.shape[0], -1)), self.fc_logvar(h.view(h.shape[0], -1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)[:, :, None, None]
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DoomVAE(ConvVAE):
    def __init__(self, dim_C, dim_H, dim_W, z_size=64):
        super(ConvVAE, self).__init__()

        # print('[INFO] ({}) init dim_C {} dim_H {} z_size {}'.format(self.__class__.__name__,
        #                                                            dim_C, dim_W, z_size))
        self.enc = nn.Sequential(
            nn.Conv2d(dim_C, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(dim_H, 4, 2), 4, 2), 4, 2), 4, 2)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(dim_W, 4, 2), 4, 2), 4, 2), 4, 2)
        enc_out_dim = convh * convw * 256

        print('[INFO] ({}) Expected size {}'.format(self.__class__.__name__,
                                                    enc_out_dim))

        self.fc_mu = nn.Linear(enc_out_dim, z_size)
        self.fc_logvar = nn.Linear(enc_out_dim, z_size)

        self.fc_dec = nn.Linear(z_size, enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(enc_out_dim, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            # nn.ConvTranspose2d(32, dim_C, 6, stride=2), # for 64x64 image
            nn.ConvTranspose2d(32, dim_C, 4, stride=4),  ## for 120x120 image
            nn.Sigmoid()
        )


# Reconstruction + KL divergence losses summed over all elements and batch
def augmented_kl_loss(mu, logvar):
    # Augmented kl loss per dim.
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # TODO: KL tolerance?

    return KLD


def train_VAE_MNIST(epoch, model, dataloader, optimizer, log_interval, use_gpu):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        if use_gpu:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = augmented_kl_loss(mu, logvar) + F.binary_cross_entropy(recon_batch, data, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(dataloader.dataset),
                                                                           100. * batch_idx / len(dataloader),
                                                                           loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))


def test_VAE_MNIST(epoch, model, dataloader, use_gpu):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if use_gpu:
                data = data.cuda()
            recon_batch, mu, logvar = model(data)
            test_loss += augmented_kl_loss(mu, logvar).item() + F.binary_cross_entropy(recon_batch, data,
                                                                                       reduction='sum').item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'VAE_MNIST/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('VAE_MNIST/data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('VAE_MNIST/data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    os.makedirs("VAE_MNIST/results", exist_ok=True)

    model = ConvVAE(1, 28, 28)
    if use_gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train_VAE_MNIST(epoch, model, train_loader, optimizer, args.log_interval, use_gpu)
        test_VAE_MNIST(epoch, model, test_loader, use_gpu)
        with torch.no_grad():
            sample = torch.randn(64, 20)
            if use_gpu:
                sample = sample.cuda()
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), 'VAE_MNIST/results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()
