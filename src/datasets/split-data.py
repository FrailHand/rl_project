#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

import json
import numpy as np
import os

if __name__ == '__main__':
    doom_data_path = '/idiap/temp/fmarelli/hellboy/explore'

    print('[INFO] Getting files name.')
    all_files = [f for f in os.listdir(doom_data_path) if 'reward' not in f]
    L = len(all_files)

    print('[INFO] Performing data split for test(10%), validation(5%) and train (80%)')
    # split data in 85, 10 and 15 % for train, test and validation
    val_length = int(0.05 * L)
    test_length = int(0.1 * L)

    train_length = L - val_length - test_length

    all_index = np.arange(L)
    np.random.shuffle(all_index)

    # get file index
    val_index = all_index[0:val_length]
    test_index = all_index[val_length:(val_length + test_length)]
    train_index = all_index[(val_length + test_length):L]

    train_files = [all_files[idx] for idx in train_index]
    test_files = [all_files[idx] for idx in test_index]
    val_files = [all_files[idx] for idx in val_index]

    annotations = {'train_files': train_files,
                   'test_files': test_files,
                   'val_files': val_files}

    out_file = 'HELLBOY_datafolds_annotations.json'
    print('[INFO] Saving file at', out_file)
    with open(out_file, 'w') as file_:
        json.dump(annotations, file_, indent=2)
