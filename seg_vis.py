import numpy as np
import open3d as o3d


red = np.asarray([255, 0, 0])/255
green = np.asarray([0, 255, 0])/255
blue = np.asarray([0, 0, 255])/255
yellow = np.asarray([255, 255, 0])/255
light_pink = np.asarray([255, 182, 193])/255
steel_blue = np.asarray([70, 130, 180])/255
MediumSpringGreen = np.asarray([245,255,250])/255
DarkOrange = np.asarray([255,140,0])/255
Chocolate = np.asarray([210,105,30])/255
SaddleBrown = np.asarray([139,69,19])/255
Tomato = np.asarray([255,99,71])/255
SlateGray = np.asarray([112,128,144])/255
Indigo = np.asarray([75,0,130])/255
Orchid = np.asarray([218,112,214])/255
LightCyan = np.asarray([225,255,255])/255

def Match_classes_with_colors(prediction):  #prediction是一个N*1的点云分割预测结果，Tensor类型
    colors = np.zeros((prediction.size()[0],3))

    for i in range(prediction.size()[0]):  #此处可自定义类别和颜色
        if prediction[i,0] == 0: colors[i, 0], colors[i, 1], colors[i, 2] = red #ceiling
        elif prediction[i,0] == 1: colors[i, 0], colors[i, 1], colors[i, 2] = yellow    #floor
        elif prediction[i,0] == 2: colors[i, 0], colors[i, 1], colors[i, 2] = green #wall
        elif prediction[i,0] == 3: colors[i, 0], colors[i, 1], colors[i, 2] = light_pink    #beam
        elif prediction[i,0] == 4: colors[i, 0], colors[i, 1], colors[i, 2] = steel_blue    #column
        elif prediction[i,0] == 5: colors[i, 0], colors[i, 1], colors[i, 2] = MediumSpringGreen #window
        elif prediction[i,0] == 6: colors[i, 0], colors[i, 1], colors[i, 2] = DarkOrange    #door
        elif prediction[i,0] == 7: colors[i, 0], colors[i, 1], colors[i, 2] = Chocolate #table
        elif prediction[i,0] == 8: colors[i, 0], colors[i, 1], colors[i, 2] = blue   #chair
        elif prediction[i,0] == 9: colors[i, 0], colors[i, 1], colors[i, 2] = Tomato    #sofa
        elif prediction[i,0] == 10: colors[i, 0], colors[i, 1], colors[i, 2] = SlateGray    #bookcase
        elif prediction[i,0] == 11: colors[i, 0], colors[i, 1], colors[i, 2] = Indigo   #board
        elif prediction[i,0] == 12: colors[i, 0], colors[i, 1], colors[i, 2] = Orchid   #clutter
        elif prediction[i,0] == 13: colors[i, 0], colors[i, 1], colors[i, 2] = LightCyan
        
    
    return colors


def Visualize_Colored_PointCloud(points, colors):
    pcd = o3d.PointCloud()  #定义open3d的PointCloud对象
    pcd.points = o3d.Vector3dVector(points)  
    pcd.colors = o3d.Vector3dVector(colors)
    o3d.draw_geometries([pcd])