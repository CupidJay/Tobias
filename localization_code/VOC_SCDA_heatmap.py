import os
import sys
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from skimage import measure
# from scipy.misc import imresize
from utils.func import *
from utils.vis import *
from utils.IoU import *
import json

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"
cudnn.benchmark = True

#pretrained = 'random'
#pretrained = 'resnet50'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_800ep_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v1_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_1200ep_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_112x112_pretrain_200ep_lr_0.03_224x224.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl_checkpoints/simclr_cub_800ep_bs_128_lr_0.125_224x224_pretrain.pth.tar'



prefix = 'random_heatmap'

model_name = 'resnet50'
#prefix = 'up2'

#model = models.resnet50(pretrained=False)
#print(model)
#model = load_pretrained_model(model, pretrained)
#removed = list(model.children())[:-2]
#model = torch.nn.Sequential(*removed)


#model = models.vgg16(pretrained=False)
#print(model)
#model = load_pretrained_model(model, pretrained)
#model = model.features

if model_name == 'resnet50':
    model = models.resnet50(pretrained=False)
    #print(model)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'vgg16':
    model = models.vgg16(pretrained=False)
    #print(model)
    model = model.features

#model = models.UNet()
model = model.cuda()

imagedir = '/opt/Dataset/VOCdevkit/VOC2007/JPEGImages'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=(1.0 / 255, 1.0 / 255, 1.0 / 255),
                                 std=[0.229, 0.224, 0.225]
                                 )

transform = transforms.Compose([
    # transforms.Resize((h, w)),
    transforms.ToTensor(),
    normalize
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(1.0 / 255, 1.0 / 255, 1.0 / 255))
    # transforms.Normalize(mean=(123.68,116.78,103.94),std=(1,1,1))
])

files = os.listdir(imagedir)
files.sort()

for (i, name) in enumerate(files):
    #raw_img = cv2.imread(os.path.join(imagedir, name))
    #h, w, _ = raw_img.shape

    raw_img = Image.open(os.path.join(imagedir, name)).convert('RGB')
    w, h = raw_img.size

    #raw_img = cv2.resize(raw_img,(h, w))
    #raw_img = raw_img.astype(np.float32)

    #raw_img[:,:,0] = raw_img[:,:,0] - 103.94
    #raw_img[:,:,1] = raw_img[:,:,1] - 116.78
    #raw_img[:,:,2] = raw_img[:,:,2] - 123.68
    img = transform(raw_img)
    img = torch.unsqueeze(img, 0)

    #img = torch.from_numpy(raw_img.transpose(2, 0, 1))
    #img = torch.unsqueeze(img, 0)
    img = to_variable(img)

    vis = model(img)
    vis = to_data(vis)

    # DDT
    vis = torch.squeeze(vis)
    vis = vis.numpy()

    vis = np.transpose(vis, (1, 2, 0))
    he, we, ce = vis.shape

    vis_sum = np.sum(vis, axis=2)
    #vis_mean = np.mean(vis_sum)

    #print(vis_sum.shape, vis_mean)

    #highlight = np.zeros((he, we))
    #highlight[vis_sum > vis_mean] = 1

    mask = (vis_sum - np.min(vis_sum)) / np.max(vis_sum)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)


    # visualize heatmap
    # show highlight in origin image
    if i<500:
        savepath = 'VOC/Visualization/SCDA/{}/{}'.format(model_name, prefix)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        raw_img = cv2.imread(os.path.join(imagedir, name))

        res = heatmap + np.float32(raw_img)
        res = res / np.max(res)
        res = np.uint8(255 * res)

        #cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]), (temp_bbox[2]+temp_bbox[0], temp_bbox[3]+temp_bbox[1]), (0,255,0), 4)
        cv2.imwrite(os.path.join(savepath, str(name)+'.jpg'), np.asarray(res))
    else:
        break