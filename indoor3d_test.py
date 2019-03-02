import argparse
import numpy as np
import torch
import torch.utils.data
from indoor3d_dataset import S3DIS
from pointnet.model import PointNetDenseCls
from torch.autograd import Variable
import open3d as o3d
from seg_vis import *

parser = argparse.ArgumentParser()


parser.add_argument('--model', type=str, default='log6/seg_model_49.pth', help='model path')
parser.add_argument('--area', type=str, default='Area_6', help='model index')
parser.add_argument('--index', type=int, default='200', help='data index')



opt = parser.parse_args()
print(opt)

data = S3DIS(
    root='dataset/indoor3d_sem_seg_hdf5_data',
    train=False,
    test_area='Area_6')

idx = 100

print("model %d/%d" % (idx, len(data)))

point, seg = data[idx]
print(point.size(), seg.size())

point_np = point.numpy()

#forward
num = 13
print('num={}'.format(num))
classifier = PointNetDenseCls(k=num)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _ = classifier(point)


print(pred.size())
pred_choice = pred.data.max(2)[1]
print(pred_choice.size())
print(point.size())
# print(point[0, :, 0])
point = point.squeeze()
# print(point[:, 0])
point = point.transpose(1, 0)
# print(point[0, :])
print(point.size())
pred_choice = pred_choice.transpose(1, 0)
print(pred_choice.size())

colors = Match_classes_with_colors(pred_choice)

Visualize_Colored_PointCloud(point, colors)
