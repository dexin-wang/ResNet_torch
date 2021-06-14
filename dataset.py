'''
Description: 数据集
Author: wangdx
Date: 2021-06-12 18:07:38
LastEditTime: 2021-06-13 15:29:58
'''
from posixpath import join
import numpy as np
import torch
import os
import cv2
import glob


class Image:
    def __init__(self, file) -> None:
        # print(file)
        self.img = cv2.imread(file)
    
    def resize(self, size=(224,224)):
        """
        直接缩放成size，不等比例缩放
        size: tuple(w,h)
        """
        self.img = cv2.resize(self.img, size)
    
    def normalise(self):
        """
        归一化
        除以255, 减均值
        """
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

    


class Dataset:
    def __init__(self, path, split, mode) -> None:
        """
        path: 数据集路径 
        split: float [0, 1] 训练集和验证集的分割比例，0-split为训练集，split-1为验证集
        mode: 'train' or 'val'
        """
        self.path = path
        
        # 分别读取cat和dog图像列表
        cat_list = glob.glob(os.path.join(path, 'Cat', '*.jpg'))
        dog_list = glob.glob(os.path.join(path, 'Dog', '*.jpg'))

        # 分割训练集和验证集
        split_n = int(len(cat_list)*split)
        if mode == 'train':
            cat_list = cat_list[:split_n]
            dog_list = dog_list[:split_n]
        else:
            cat_list = cat_list[split_n:]
            dog_list = dog_list[split_n:]
        
        # 生成标签
        cat_labels = [0] * len(cat_list)
        dog_labels = [1] * len(dog_list)

        # 合并
        self.img_files = cat_list + dog_list
        self.targets = cat_labels + dog_labels


    def __len__(self):
        return len(self.img_files)
    

    def __getitem__(self, idx):
        img_data = Image(self.img_files[idx])
        img_data.resize()
        # img_data.normalise()

        img_np = img_data.img.transpose((2, 0, 1))  # (H,W,C)->(C,H,W)
        img_tensor = self.np_to_tensor(img_np)  # img数据
        target_tensor = torch.tensor([self.targets[idx]])   # label

        return img_tensor, target_tensor

    
    def np_to_tensor(self, data):
        return torch.from_numpy(data.astype(np.float32))

        
    


if __name__ == '__main__':
    path = 'E:/research/dataset/classification/kaggle_cat_dog/PetImages'
    dataset = Dataset(path, 0.8, 'train')
    img, target = dataset.__getitem__(10)

    # img = img.cpu().numpy().transpose((1,2,0)).astype(np.uint8)
    
    # cv2.imshow('img', img)
    # cv2.waitKey()

    print(img.shape[0])
    print(img.size(0))
    # print(target.shape)