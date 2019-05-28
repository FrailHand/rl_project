#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import collections
import cv2
import numbers
import numpy as np


class NpCenterCrop(object):
    """Crops the given numpy array at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, array):
        c, w, h = array.shape

        crop_w = self.size[0]
        crop_h = self.size[1]
        start_w = w // 2 - crop_w // 2
        start_h = h // 2 - crop_h // 2
        return array[:, start_w:start_w + crop_w, start_h:start_h + crop_h]


class NpResize3d(object):
    """Resize the input 3d numpy array to the given size in second and third dimension.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired cv2 interpolation.
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, array):

        if isinstance(self.size, int):
            w, h = array.shape[1:]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return array
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return self._resize_3d(array, ow, oh, self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return self._resize_3d(array, ow, oh, self.interpolation)
        else:
            # self.size is a tuple.
            return self._resize_3d(array, *self.size, self.interpolation)

    @staticmethod
    def _resize_3d(array, out_w, out_h, interpolation):
        out_array = np.empty((array.shape[0], out_h, out_w), dtype=array.dtype)
        for idx in range(len(array)):
            out_array[idx] = cv2.resize(array[idx], (out_w, out_h), interpolation)
        return out_array
