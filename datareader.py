# title       :datareader
# description :Script to extract and load the 4 datasets needed for our papers
#              This script is extendable and can accommodate many more datasets
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Either execute the file with "dataset_name" and "dataset_path" specified or call it in utils.py.
# notes       :Uses already downloaded datasets to prepare them for our models
import csv
import glob
import sys

import h5py
import numpy as np
import pandas as pd
import simplejson as json


# Structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data
class DataReader:
    def __init__(self, dataset, datapath, _type='original'):
        if dataset == 'daphnet':
            self.data, self.idToLabel = self._read_daphnet(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        else:
            print('Dataset is not yet supported!')
            sys.exit(0)

    def save_data(self, dataset, path=""):
        f = h5py.File(f'{path}{dataset}.h5', mode='w')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open(f'{path}{dataset}.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.idToLabel))
        print('Done.')

    @property
    def train(self):
        return self.data['train']

    @property
    def validation(self):
        return self.data['validation']

    @property
    def test(self):
        return self.data['test']

    def _read_daphnet(self, datapath):
        #         files = {
        #     'train': [
        #         'S01R01.txt', 'S01R02.txt',
        #         'S02R01.txt', 'S02R02.txt',
        #         'S03R01.txt', 'S03R02.txt',
        #         'S03R03.txt', 'S05R01.txt',
        #         'S05R02.txt',
        #         'S06R01.txt', 'S06R02.txt',
        #         'S07R01.txt', 'S07R02.txt',
        #         'S09R01.txt'
        #     ],
        #     'validation': [
        #         'S08R01.txt'
        #     ],
        #     'test': [
        #         'S08R01.txt'
        #     ]
        # }
        files = {
            'train': [
                'S01R01.txt', 'S01R02.txt',
                'S03R01.txt', 'S03R02.txt',
                'S06R01.txt', 'S06R02.txt',
                'S07R01.txt', 'S07R02.txt',
                'S08R01.txt', 'S09R01.txt', 'S10R01.txt'
            ],
            'validation': [
                'S02R02.txt', 'S03R03.txt', 'S05R01.txt'
            ],
            'test': [
                'S02R01.txt', 'S04R01.txt', 'S05R02.txt'
            ]
        }
        label_map = [
            # (0, 'Other')
            (1, 'No freeze'),
            (2, 'Freeze')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = {dataset: self._read_daph_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_daph_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the non related activity
                    if line[10] == "0":
                        continue
                    for ind in cols:
                        if ind == 10:
                            if line[ind] == "0":
                                continue
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     _dataset = sys.argv[1]
    #     _datapath = sys.argv[2]
    # else:
    #     _dataset = input('Enter Dataset name e.g. opportunity, daphnet, ucihar, pamap2:')
    #     _datapath = input('Enter Dataset root folder: ')
    # print(f'Reading {_dataset} from {_datapath}')
    dr = DataReader("daphnet", "daphnet")
