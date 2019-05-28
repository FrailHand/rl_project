#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import numpy as np
import os
# always torch after cv2
import torch.utils.data


class DoomDataset(torch.utils.data.Dataset):
    """Doom dataset definition"""

    def __init__(self, ipath, data_fold, transform=None, one_hot=False):
        self.ipath = ipath
        self.data_fold = data_fold
        self.do_verbose = True
        self.transform = transform
        self.one_hot = one_hot

        self.data = None
        self.size = None

        self.num_frames = 2100
        self.num_label_classes = 200

        self.load_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # find range lying in index
        sample = self.data[idx, :]
        # print(np.unique(sample))

        if self.one_hot:
            sample_one_hot = DoomDataset.to_one_hot(sample[1], one_hot_length=self.num_label_classes).reshape(
                self.num_label_classes, *sample.shape[1:])
            sample = np.concatenate((sample[0:1], sample_one_hot), axis=0)

            # batch_idx = (np.ones(label_idx.shape)*i).astype(int)
            # x_idxs = (np.arange(label_idx.size) / 160).astype(int).reshape(label_idx.shape)
            # y_idxs = (np.arange(label_idx.size) % 160).astype(int).reshape(label_idx.shape)
            # other_exp[batch_idx, label_idx, x_idxs, y_idxs] = 1.0

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def load_data(self, max_exp_files=10):
        # number of examples (labels, depth, action, reward)
        self.data = None

        exp_files = [os.path.join(self.ipath, f) for f in self.data_fold]

        offset = 0
        print('[INFO] (%s) Loading data into memory' % self.__class__.__name__)

        pointers = [None] * len(exp_files)

        for e in range(len(exp_files)):
            explore = np.load(exp_files[e])

            # load original file in shape (20, 2100, 2, 120, 160)
            # reshape to (2100*20, 2, 120, 160)
            # other_exp= np.stack([explore[k].astype(np.float32) for k in explore.keys()])
            other_exp = np.empty((20, self.num_frames, 2, 120, 160), np.float32)
            for i, v in enumerate(explore.values()):
                other_exp[i] = v[:self.num_frames].astype(np.float32)
                # other_exp[i] = v[:self.num_frames]

            other_exp = np.reshape(other_exp, (-1, 2, 120, 160))
            other_exp = self.prune_data(other_exp)

            # print('[INFO] (%s) The shape is %s' % (self.__class__.__name__, other_exp.shape))
            pointers[e] = other_exp

        self.data = np.concatenate(pointers, axis=0)
        print('[INFO] (%s) The dimension of data is %s with %f GB'
              % (self.__class__.__name__, self.data.shape, self.data.nbytes / 1000000000.))

        self.size = self.data.shape[0]

    def prune_data(self, chunk):
        print('[INFO] ({}) Prunning data according to labels!'.format(self.__class__.__name__))
        # count of self in the image is around
        print('[INFO] Shape of old data', chunk.shape)
        accepted_idx = []
        for i in range(chunk.shape[0]):
            label = chunk[i, 0, ...]

            # background or self
            nozero = np.where((label > 0) & (label < 255))
            # selfcount= np.where(label==255)

            # take 75% of the times images when there are no objects!
            if len(nozero[0]) == 0:
                if np.random.random() > 0.25:
                    accepted_idx.append(i)
            else:
                accepted_idx.append(i)

            # print('[INFO] The number of nonzero and non self is', len(nozero[0]))
            # print('[INFO] The percentage is', len(nozero[0])/float(label.size))
            # print('[INFO] The percentage of self is', len(selfcount[0])/float(label.size))

        new_data = chunk[accepted_idx, ...].copy()
        print('[INFO] Shape of new data', new_data.shape)
        del chunk

        return new_data

    @staticmethod
    def to_one_hot(a, one_hot_length=None):
        if one_hot_length is None:
            one_hot_length = a.max() + 1

        out = np.zeros((a.size, one_hot_length), dtype=np.uint8)

        out[np.arange(a.size), a.astype(int).ravel()] = 1
        out.shape = a.shape + (one_hot_length,)
        return out
