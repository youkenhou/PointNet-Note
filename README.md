# 记录PointNet PyTorch版本的学习
PyTorch版本PointNet作者Tensorflow版本页面上给出的github地址，[这里](https://github.com/fxia22/pointnet.pytorch)

## 数据集
PyTorch版本只给出了ModelNet的数据，只能进行部件分割，但是我想测试S3DIS数据集，所以下载了原作者给的链接中已经处理好的数据，文件为h5格式。
作者对所有的点云进行了采样，每个采样空间是一个立方体，做成一个数据，每个数据有4096个点；一个h5文件中是`1000*4096*9`个数字，代表1000个点云，每个点云中有4096个点，每个点有9个值`xyz`，`rgb`，剩下三个还不知道。
具体处理过程在Tensorflow版本中给出了，太复杂看不太懂。
利用这些处理好的h5文件，结合Tensorflow版本的代码写出PyTorch的数据集class，代码在`indoor3d_dataset.py`中。

## 训练
训练的代码基本参考了PyTorch版本的，只是将刚开始数据集的读取改成了S3DIS的，代码在`train_indoor_3d.py`中。

## 结果可视化
PyTorch版本用了原作者Tensorflow版本中提供的可视化代码，用opencv写的，跑不起来。
利用Open3D这个Python包写了另一种可视化方法，在`indoor3d_test.py`和`seg_vis`中，`indoor3d_test.py`包含了test代码，参考了PyTorch版本的test代码，只是把可视化部分换掉了。
