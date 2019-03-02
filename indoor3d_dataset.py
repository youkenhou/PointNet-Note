from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import h5py

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][...,0:3]
    label = f['label'][:]
    return (data, label)

class S3DIS(data.Dataset):
    def __init__(self, root, train=True, test_area=None):
        self.root = root
        self.train = train
        
        all_files = getDataFiles(os.path.join(root, 'all_files.txt'))   #所有的h5文件名
        room_filelist = getDataFiles(os.path.join(root, 'room_filelist.txt'))   #所有的房间名字，e.g. Area_1_hallway_1
        
        data_batch_list = []    #把所有h5文件中的data都存到这里
        label_batch_list = []   #把所有h5文件中的label都存到这里
        for h5_filename in all_files:
            data_batch, label_batch = load_h5('dataset/' + h5_filename)
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)   #所有data连接
        label_batches = np.concatenate(label_batch_list, 0) #所有label连接

        # print(data_batches.shape)
        # print(label_batches.shape)

        self.test_area = test_area  #测试集的区域名称，e.g. Area_6
        
        train_idxs = []
        test_idxs = []
        for i,room_name in enumerate(room_filelist):
            if test_area in room_name:  #如果房间名中包含这个区域名，说明这个是测试集的区域，将index记录到test_idxs中
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if self.train:  #根据需要训练集还是测试集来进行选择
            self.points = data_batches[train_idxs,...]
            self.labels = label_batches[train_idxs,...]
        else:
            self.points = data_batches[test_idxs,...]
            self.labels = label_batches[test_idxs,...]

    def __getitem__(self, index):
        pt_idxs = np.arange(0, 4096)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[index, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[index, pt_idxs].copy()).type(
            torch.LongTensor
        )

        return current_points, current_labels
    
    def __len__(self):
        return self.points.shape[0]

if __name__ == '__main__':
    data_path = 'dataset/indoor3d_sem_seg_hdf5_data'
    test_area = 'Area_6'
    data_set = S3DIS(data_path, train=True, test_area=test_area)
