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
import custom_models

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"
cudnn.benchmark = True

input_size = 224

def copy_parameters(model, pretrained):
    model_dict = model.state_dict()
    pretrained_dict = pretrained.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].size()==model_dict[k].size()}

    for k, v in pretrained_dict.items():
        print(k)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


#pretrained = 'random'
#pretrained = 'resnet50'
#pretrained = './resnet50_pixel.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_800ep_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v1_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_200ep_224x224_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/random_potential/checkpoints/moco_finetune/scda_r50_moco_cub_200ep_224x224_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_112x112_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/simclr/checkpoints/simclr_cub_200ep_bs_128_lr_0.125_224x224_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/random_potential/checkpoints/simclr_finetune/scda_r50_simclr_cub_200ep_bs_128_lr_0.125_224x224_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/cub_checkpoints/simclr_800ep_224_ft.pth.tar'
#pretrained = 'random'
#pretrained = 'checkpoints/two_branch_update/median/finetune/224/custom_resnet50_custom_vgg16/gpus_4_lr_0.001_bs_128_epochs_50_path_cub200/custom_resnet50_checkpoint_0010.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl_checkpoints/simclr_cub_800ep_bs_512_lr_0.5_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_1200ep_56x56_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_56x56_pretrain_200ep_lr_0.03_112x112v2_100ep_lr_0.03_224x224v2.pth.tar'
#pretrained = 'checkpoints/mask_GAPv2/custom_resnet50_mask_gap_lr_0.001_bs_64_epochs_50/checkpoint.pth.tar'
#import random
#random seed
'''
seed = 0
np.random.seed(seed)
#random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

prefix = 'random_seed/{}'.format(seed)
'''

prefix = 'random2'
model_name = 'uniform_resnet50'

if 'random' in prefix:
    flag = False
else:
    flag = True

if model_name == 'resnet50':
    model = custom_models.resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'kaiming_uniform_resnet50':
    model = custom_models.kaiming_uniform_resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'xavier_resnet50':
    model = custom_models.xavier_resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'xavier_uniform_resnet50':
    model = custom_models.xavier_uniform_resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'uniform_resnet50':
    model = custom_models.uniform_resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'gauss_resnet50':
    model = custom_models.gauss_resnet50(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_conv1':
    model = custom_models.resnet50_conv1(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_conv1_2':
    model = custom_models.resnet50_conv1_2(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_conv1_3':
    model = custom_models.resnet50_conv1_3(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_conv1_4':
    model = custom_models.resnet50_conv1_4(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_sigmoid':
    model = custom_models.resnet50_sigmoid(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet50_arctan':
    model = custom_models.resnet50_arctan(pretrained=False)
    if flag:
        pretrained = models.resnet50(pretrained=True)
        model = copy_parameters(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'resnet18':
    model = models.resnet18(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    removed = list(model.children())[:-2]
    model = torch.nn.Sequential(*removed)
elif model_name == 'vgg16':
    model = models.vgg16(pretrained=flag)
    #print(model)
    #model = load_pretrained_model(model, pretrained)
    model = model.features
#model = models.UNet()
print(model)

model.eval()
model = model.cuda()

root = '/opt/Dataset/CUB_200_2011/CUB_200_2011'
imagedir = os.path.join(root, 'images')

name_bbox_dict = get_bbox_dict(root)

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


classes = os.listdir(imagedir)
classes.sort()

acc = []
for cls in classes:
    print('*' * 15)
    print(cls)
    total = 0
    IoUSet = []

    files = os.listdir('/opt/caoyh/datasets/cub200/val/%s'%cls)
    files.sort()

    for (i, name) in enumerate(files):
        gt_bbox = name_bbox_dict[os.path.join(cls, name)]

        #raw_img = cv2.imread(os.path.join(imagedir, cls, name))
        #h, w, _ = raw_img.shape

        raw_img = Image.open(os.path.join(imagedir, cls, name)).convert('RGB')
        w, h = raw_img.size

        rateh = input_size/h
        ratew = input_size/w

        w = input_size
        h = input_size
        raw_img = raw_img.resize((w, h))

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

        #print(vis.shape)

        vis = np.transpose(vis, (1, 2, 0))
        he, we, ce = vis.shape

        vis_sum = np.sum(vis, axis=2)
        vis_mean = np.mean(vis_sum)
        #vis_mean = np.percentile(vis_sum, 60)
        #print(vis_sum.shape, vis_mean)

        highlight = np.zeros((he, we))
        highlight[vis_sum > vis_mean] = 1

        # max component
        all_labels = measure.label(highlight)
        highlight = np.zeros(highlight.shape)
        highlight[all_labels == count_max(all_labels.tolist())] = 1

        # visualize heatmap
        # show highlight in origin image
        highlight = np.round(highlight * 255)
        # highlight_big = imresize(Image.fromarray(np.uint8(highlight)), (h, w), interp='nearest')
        # highlight_big = Image.fromarray(np.uint8(highlight)).resize((w, h))
        # props = measure.regionprops(highlight_big)
        highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
        props = measure.regionprops(highlight_big.astype(int))

        if len(props) == 0:
            print(highlight)
            bbox = [0, 0, w, h]
        else:
            temp = props[0]['bbox']
            bbox = [temp[1], temp[0], temp[3], temp[2]]

        gt_bbox[0] = int(gt_bbox[0] * ratew)
        gt_bbox[1] = int(gt_bbox[1] * rateh)

        gt_bbox[2] = int(gt_bbox[2] * ratew)
        gt_bbox[3] = int(gt_bbox[3] * rateh)

        gt_bbox[2] = gt_bbox[2]+gt_bbox[0]
        gt_bbox[3] = gt_bbox[3]+gt_bbox[1]

        iou = IoU(bbox, gt_bbox)


        # print(max_iou)
        IoUSet.append(iou)

        temp_bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        #ddt_bbox[os.path.join(cls, name)] = temp_bbox

        #'''
        highlight_big = np.expand_dims(np.asarray(highlight_big), 2)
        highlight_3 = np.concatenate((np.zeros((h,w,1)), np.zeros((h, w,1))), axis=2)
        highlight_3 = np.concatenate((highlight_3, highlight_big), axis=2)

        if i<1:
            savepath = 'CUB/Visualization/SCDA_PIL/{}/{}'.format(model_name, prefix)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            #raw_img = cv2.imread(os.path.join(imagedir, name))
            #raw_img = cv2.resize(raw_img, (h, w))

            raw_img = cv2.cvtColor(np.asarray(raw_img), cv2.COLOR_RGB2BGR)
            #cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]), (temp_bbox[2]+temp_bbox[0], temp_bbox[3]+temp_bbox[1]), (0,255,0), 4)
            cv2.imwrite(os.path.join(savepath, str(name)+'.jpg'), np.asarray(raw_img)+highlight_3)

        #'''
    cls_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    print('{} corLoc acc is {}'.format(cls, cls_acc))
    acc.append(cls_acc)

print(acc)
print(model_name, prefix)
print(np.mean(acc))