'''
Description: 训练脚本，本程序按照最简单的形式编程，以方便理解
Author: wangdx
Date: 2021-06-12 18:06:46
LastEditTime: 2021-06-14 14:17:27
'''

import torch
import torch.utils.data
import torch.optim
import tensorboardX
from torchsummary import summary
import torchsummary

from dataset import Dataset
from model import ResNet_34, ResNet_50, compute_loss, evaluation

import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)



def train(epoch, net, train_data, optimizer):
    """
    训练函数
    epoch: 当前epoch
    net: 网络模型
    train_data: 训练集加载器
    optimizer: 优化器
    """
    ret = {'loss': 0}

    net.train()
    batch_sum = len(train_data)
    batch_num = 0
    for x, target in train_data:
        pred = net(x)
        loss = compute_loss(pred, target)
        
        batch_num += 1
        logging.info('Epoch: {}, Batch: {}/{}, loss: {:.5f}'.format(epoch, batch_num, batch_sum, loss.item()))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        ret['loss'] += loss.item() / batch_sum

    return ret


def validate(net, val_data):
    """
    net: 网络模型
    val_data: 验证集加载器
    """
    net.eval()

    ret = {
        'loss': 0, 
        'fail': 0, 
        'success': 0,
        'acc': 0
    }
    
    batch_num = len(val_data)
    batch = 0

    with torch.no_grad():
        for x, target in val_data:
            pred = net(x)
            loss = compute_loss(pred, target)

            batch_size = target.shape[0]
            ret['loss'] += loss.item() / (batch_num * batch_size)
            
            success_num = evaluation(pred, target)
            ret['success'] += success_num
            ret['fail'] += batch_size - success_num

            batch += 1
            print('\rValidating... {:.2f}'.format(batch/batch_num), end='')

    ret['acc'] = ret['success'] / (ret['success'] + ret['fail'])
    return ret



def run():
    
    # 加载数据集
    logging.info('loading dataset ...')
    path = 'E:/research/dataset/classification/kaggle_cat_dog/PetImages'
    train_dataset = Dataset(path, split=0.005, mode='train')
    train_data = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True)
    
    val_dataset = Dataset(path, split=0.995, mode='val')
    val_data = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=8)

    # 加载网络
    logging.info('building network ...')
    net = ResNet_50()

    # 设置优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 初始化tensorboard
    desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir = './output'
    tb_path = os.path.join(output_dir, 'tensorboard', desc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = tensorboardX.SummaryWriter(tb_path)

    # 迭代训练
    logging.info('start train')
    epochs = 100
    val_acc = 0
    for epoch in range(epochs):
        # 训练
        train_ret = train(epoch, net, train_data, optimizer)

        tb.add_scalar('train/loss', train_ret['loss'], epoch)

        # 每训练一定次数，验证并保存模型
        if epoch % 5 == 0:
            logging.info('evaluating...')

            val_ret = validate(net, val_data)
            tb.add_scalar('val/loss', val_ret['loss'], epoch)
            tb.add_scalar('val/accuracy', val_ret['acc'], epoch)
            
            # 精度提升时，保存模型
            if val_ret['acc'] >= val_acc:
                val_acc = val_ret['acc']
                model_path = os.path.join(output_dir, 'model', desc)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                model_name = 'epoch_{}_acc_{:.3f}'.format(epoch, val_acc)
                logging.info('\nsave model: ' + model_name)
                torch.save(net, os.path.join(model_path, model_name))



if __name__ == '__main__':
    run()