import torch
import os
import yaml
from matplotlib import pyplot as plt
import numpy as np
import math
import transforms
from data import dataset
import model
import cv2
import utils
import time
from datetime import datetime

def acc_grad(net, criterion, optimizer, data_collects, device):
    start_time = time.time()
    counter = 0
    acc_loss = 0
    optimizer.zero_grad()
    for data_item in data_collects:
        images = data_item['images'].to(device)
        labels = data_item['labels'].to(device)
        prediction = net(images)
        loss = criterion(prediction, labels)
        loss.backward()
        counter += 1
        acc_loss += loss.item()
    optimizer.step()
    acc_loss = acc_loss / counter
    duration = time.time() - start_time
    return duration, acc_loss

def trainer(cfgs, train_dataset):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=0)

    # model
    net = model.DRNet(cfgs).train()
    # loss
    criterion = model.Loss(cfgs)
    # optimal
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}], weight_decay=cfgs['weight_decay'])
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                     lr=cfgs['lr'], momentum=cfgs['momentum'], weight_decay=cfgs['weight_decay'])
    # GPU
    device = torch.device(cfgs['device'] if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion.to(device)

    # train
    for epoch in range(cfgs['max_epoch']):  # loop over the dataset multiple times
        model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
        running_loss = 0.0
        data_collects = []
        idx = 0
        for i, data in enumerate(dataloader, 0):
            data_collects.append(data)
            if len(data_collects) == cfgs['acc_grad'] or i == train_dataset.length-1:
                duration, loss = acc_grad(net, criterion, optimizer, data_collects, device)
                data_collects.clear()
                idx += cfgs['batch_size']
                running_loss += loss
            else:
                continue

            print_epoch = 10 * cfgs['acc_grad']
            if i % print_epoch == print_epoch - 1:
                examples_per_sec = 10 / duration
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d], loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), epoch + 1, i + 1, running_loss / 10,
                                    examples_per_sec, sec_per_batch))
                running_loss = 0.0

    save_name = utils.cfgs2name(cfgs)
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    torch.save(net.state_dict(), './' + save_name + '/' + save_name + '.pth')
    print('Finished Training')



    

