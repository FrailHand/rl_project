#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import sys

import argparse
import json
import os
import torch
from functools import partial
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Train VAE on saved VizDoom explorations.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output', metavar='DIR', type=str, dest="out_dir",
                    help="output directory to store network and images", default="VAE_Doom")
parser.add_argument('--path', metavar='DIR', type=str,
                    help="Root of the project. Has to be added to system path to allow imports on grid.")

args = parser.parse_args()
if args.path is not None:
    sys.path.append(args.path)

# Local source tree imports.
from src.datasets.doom_dataset import DoomDataset
from src.datasets.data_transformer import *
from src.models.vae import DoomVAE, augmented_kl_loss

doom_input_w = 120
doom_input_h = 120

HELLBOY_annotations = '/idiap/temp/amartinez/src/hellboy/src/annotations/HELLBOY_datafolds_annotations.json'
data_path = '/idiap/temp/fmarelli/hellboy/explore'
# data_path = 'src/annotations/explore/'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_vae(epoch, model, annotations, optimizer, log_interval=10, scheduler=None, training=True, transform=None):
    # Generate data partitions containing 6 files, 10GB approx. Number of chunks is arbitrary.
    n_exploration_files = len(annotations)
    n_files_per_data_set = 4
    n_chunks = n_exploration_files // n_files_per_data_set
    data_chunks = [np.random.choice(annotations, size=n_files_per_data_set, replace=True).tolist() for i in
                   range(n_chunks)]

    # kwargs = {'num_workers': 5, 'pin_memory': True}  # if device=='cuda' else {}

    total_loss = 0
    chunk_idx = 0
    label_weight = 3.0
    for chunk_idx, chunk in enumerate(data_chunks):
        # Generate data set for given chunk.
        doom_dataset = DoomDataset(data_path, chunk, transform, one_hot=False)
        data_loader = torch.utils.data.DataLoader(doom_dataset,
                                                  batch_size=args.batch_size,
                                                  drop_last=False,
                                                  shuffle=True,
                                                  num_workers=4)

        for batch_idx, data in enumerate(data_loader):
            # print('[INFO] Data size', data.size())
            # data.to(device)
            data = data.cuda()

            recon_batch, mu, logvar = model(data)
            # print('Reconstructed batch size', recon_batch.size())

            kl_loss = augmented_kl_loss(mu, logvar)
            depth_loss = F.mse_loss(recon_batch[:, -1], data[:, -1], reduction='sum')
            # label_loss = torch.zeros((1,), requires_grad=False)
            #  previous loss
            # label_loss = F.mse_loss(recon_batch[:, 0], data[:, 0], reduction='sum')
            label_loss = F.smooth_l1_loss(recon_batch[:, 0], data[:, 0], reduction='sum')

            # label_loss = F.l1_loss(recon_batch[:, 0], data[:, 0], reduction='sum')
            # label_loss = F.cross_entropy(recon_batch[:, 0:-1], data[:, 0].argmax(axis=2, keepdim=True), reduction='sum')
            loss = kl_loss + depth_loss + label_weight * label_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(
                        'Train Epoch: {} Chunk: [{}/{}]  [{}/{} ({:.0f}%)]\tLoss: KL={:.6f} depth={:.6f} labels={:.6f}'
                            .format(epoch,
                                    chunk_idx + 1,
                                    len(data_chunks),
                                    batch_idx * len(data),
                                    len(data_loader.dataset),
                                    100. * batch_idx / len(data_loader),
                                    kl_loss.item() / len(data),
                                    depth_loss.item() / len(data),
                                    label_loss.item() / len(data)))
                    # if scheduler is not None:
                    #     scheduler.step(loss.item())
            else:
                if batch_idx % log_interval == 0:
                    print('Test Chunk: [{}/{}]  [{}/{} ({:.0f}%)]\tLoss: KL={:.6f} depth={:.6f} labels={:.6f}'
                          .format(chunk_idx + 1,
                                  len(data_chunks),
                                  batch_idx * len(data),
                                  len(data_loader.dataset),
                                  100. * batch_idx / len(data_loader),
                                  kl_loss.item() / len(data),
                                  depth_loss.item() / len(data),
                                  label_loss.item() / len(data)))
                if batch_idx == len(data_loader):
                    depth_name = os.path.join(args.out_dir,
                                              'results',
                                              'reconstruction_depth_{}.png'.format(epoch))

                    labs_name = os.path.join(args.out_dir,
                                             'results',
                                             'reconstruction_labels_{}.png'.format(epoch))

                    n = min(data.size(0), 8)
                    comparison_depth = torch.cat(
                        [data[:n, -1:], recon_batch.view(-1, 2, doom_input_w, doom_input_h)[:n, -1:]])
                    comparison_depth /= comparison_depth.max()
                    comparison_labels = torch.cat(
                        [data[:n, 0:1], recon_batch.view(-1, 2, doom_input_w, doom_input_h)[:n, 0:1]])
                    save_image(comparison_depth.cpu(), depth_name, nrow=n)
                    save_image(comparison_labels.cpu(), labs_name, nrow=n)

            total_loss += loss.item()

        model_name = os.path.join(args.model_path, 'doom_vae_chunk_{:08d}.nn'.format(chunk_idx))
        torch.save(model.state_dict(), model_name)

        total_loss /= len(data_loader)
        del data_loader
        del doom_dataset

    total_loss /= chunk_idx + 1
    return total_loss


def train_vae(epoch, model, annotations, optimizer, scheduler, log_interval=10, transform=None):
    annotations = annotations['train_files']

    model.train()

    total_loss = run_vae(epoch, model, annotations, optimizer, log_interval, scheduler=scheduler, training=True,
                         transform=transform)
    print('====> Train set loss: {:.4f}'.format(total_loss))


def test_vae(epoch, model, annotations, optimizer, transform=None):
    annotations = annotations['test_files']

    model.eval()
    with torch.no_grad():
        total_loss = run_vae(epoch, model, annotations, optimizer, training=False, transform=transform)
    print('====> Test set loss: {:.4f}'.format(total_loss))


# def plot_vae(num_samples, model, annotations, use_gpu=False, transform=None):
#    annotations = annotations['test_files']
#
#    data_chunks = [np.random.choice(annotations, size=1, replace=True).tolist()]
#
#    kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}
#    batch_size = min(args.batch_size, num_samples)
#
#    for chunk_idx, chunk in enumerate(data_chunks):
#        # Generate data set for random chunk.
#        doom_dataset = DoomDataset(data_path, chunk, transform, one_hot=False)
#        data_loader = torch.utils.data.DataLoader(doom_dataset,
#                                                  batch_size=batch_size,
#                                                  drop_last=False,
#                                                  shuffle=True,
#                                                  **kwargs)
#
#        for batch_idx, data in enumerate(data_loader):
#            if use_gpu:
#                data = data.cuda()
#            recon_batch, mu, logvar = model(data)
#
#            # Plot stuff.
#            # save_image(data.cpu(),
#            #            os.path.join(args.out_dir, 'results', 'reconstruction_depth{}.png'.format(
#            #                "" if args.model_name is None else "_" + args.model_name)),
#            #            nrow=num_samples)
#            comparison_depth = torch.cat([data[:, -1:], recon_batch[:, -1:]])
#            comparison_depth /= comparison_depth.max()
#            comparison_labels = torch.cat([data[:, 0:1], recon_batch[:, 0:1]])
#            save_image(comparison_depth.cpu(),
#                       os.path.join(args.out_dir, 'results', 'reconstruction_depth{}.png'.format("" if args.model_name is None else "_" + args.model_name)),
#                       nrow=num_samples)
#            save_image(comparison_labels.cpu(),
#                       os.path.join(args.out_dir, 'results', 'reconstruction_labels{}.png'.format("" if args.model_name is None else "_" + args.model_name)),
#                       nrow=num_samples)
#
#            if batch_idx * batch_size >= num_samples:
#                return


def main():
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # use_gpu = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(HELLBOY_annotations) as file_:
        anns = json.load(file_)
    # os.makedirs("VAE_Doom/results", exist_ok=True)

    transform = transforms.Compose([NpCenterCrop((120, 120)),
                                    # NpResize3d((doom_input_w, doom_input_h), interpolation=cv2.INTER_CUBIC),
                                    torch.from_numpy,
                                    partial(torch.div, other=255.)])

    z_size = 1024
    print('[INFO] Training VAE with Z dim', z_size)
    print('[INFO] This VAE takes images in input {}x{}'.format(doom_input_w, doom_input_h))

    model = DoomVAE(2, doom_input_w, doom_input_h, z_size)
    # if use_gpu:
    #    model = model.cuda()

    model.to(device)

    #  prepare optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    model_path = os.path.join(args.out_dir, 'results')
    os.makedirs(model_path, exist_ok=True)
    args.model_path = model_path

    # model = torch.load(model_path + ".nn", map_location=lambda storage, loc: storage)['model']

    for epoch in range(1, args.epochs + 1):
        train_vae(epoch, model, anns, optimizer, scheduler, args.log_interval, transform)
        test_vae(epoch, model, anns, optimizer, transform)

        model_name = os.path.join(model_path, 'doom_vae_epoch_{:08d}.nn'.format(epoch))
        torch.save(model.state_dict(), model_name)
        print("Saved policy model at {}.".format(model_name))


#        with torch.no_grad():
#            sample = torch.randn(64, z_size)
#            if use_gpu:
#                sample = sample.cuda()
#            sample = model.decode(sample).cpu()
#            sample[:, -1] /= sample[:, -1].max()
#            save_image(sample.view(-1, 1, doom_input_w, doom_input_h), os.path.join(args.out_dir, 'results', 'sample_' + str(epoch) + '.png'))
#    torch.save({'model': model}, model_path + ".nn")
#    plot_vae(8, model, anns, use_gpu, transform)


if __name__ == "__main__":
    main()
