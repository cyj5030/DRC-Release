import torch
import yaml
import cv2
import os
from PIL import Image
import transforms
from matplotlib import pyplot as plt
import numpy as np
from data import dataset
import model
import utils
import time

def tester(cfgs, test_dataset):
    net = model.DRNet(cfgs).eval()
    save_name = utils.cfgs2name(cfgs)
    net.load_state_dict(torch.load('./' + save_name + '/' + save_name + '.pth'))

    device = torch.device(cfgs['device'] if torch.cuda.is_available() else "cpu")
    net.to(device)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    t_time = 0
    t_duration = 0
    name_list = test_dataset.gt_list
    length = test_dataset.length
    sv_pth = './' + save_name + '/single-scale/'
    if not os.path.exists(sv_pth):
        os.mkdir(sv_pth)
    for i, data in enumerate(dataloader):
        images = data['images'].to(device)

        star_time = time.time()
        prediction = net(images)
        prediction = prediction.cpu().detach().numpy().squeeze()
        duration = time.time() - star_time
        t_time += duration
        t_duration += 1/duration
        print('process %3d/%3d image.' % (i, length))

        cv2.imwrite(sv_pth + name_list[i] + '.png', prediction * 255)
    print('avg_time: %.3f, avg_FPS:%.3f' % (t_time / length, t_duration / length))

    #     multi
    t_time = 0
    t_duration = 0
    sv_pth = './' + save_name + '/multi-scale/'
    if not os.path.exists(sv_pth):
        os.mkdir(sv_pth)
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            images = data['images']
            width, height = data['images'].size()[2:]
            images2x = torch.nn.functional.interpolate(data['images'], scale_factor=2, mode='bilinear', align_corners=True)
            images_half = torch.nn.functional.interpolate(data['images'], scale_factor=0.5, mode='bilinear', align_corners=True)

            star_time = time.time()
            images = images.to(device)
            prediction = net(images)
            images2x = images2x.to(device)
            prediction2x = net(images2x)
            prediction2x_down = torch.nn.functional.interpolate(prediction2x, size=(width, height), mode='bilinear', align_corners=True)
            images_half = images_half.to(device)
            prediction_half = net(images_half)
            prediction_half_up = torch.nn.functional.interpolate(prediction_half, size=(width, height), mode='bilinear', align_corners=True)
            output = (prediction + prediction2x_down + prediction_half_up)/3
            output = output.cpu().detach().numpy().squeeze()
            duration = time.time() - star_time
            t_time += duration
            t_duration += 1/duration
            print('process %3d/%3d image.' % (i, length))
            cv2.imwrite(sv_pth + name_list[i] + '.png', output*255)
    print('avg_time: %.3f, avg_FPS:%.3f' % (t_time/length, t_duration/length))
    
