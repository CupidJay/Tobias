import os
import sys
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from skimage import measure
from scipy.ndimage import label
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

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
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
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_800ep_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v2_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/official_checkpoints/moco_v1_200ep_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_1200ep_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_112x112_pretrain_200ep_lr_0.03_224x224.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl_checkpoints/simclr_cub_800ep_bs_128_lr_0.125_224x224_pretrain.pth.tar'

#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/cub_checkpoints/simclr_800ep_224_ft.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl_checkpoints/simclr_cub_800ep_bs_512_lr_0.5_112x112_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_1200ep_56x56_pretrain.pth.tar'
#pretrained = '/opt/caoyh/code/SSL/cub_from_scratch/ssl/checkpoints/res50_cub_800ep_56x56_pretrain_200ep_lr_0.03_112x112v2_100ep_lr_0.03_224x224v2.pth.tar'

prefix = 'random2'
model_name = 'resnet50'

if 'random' in prefix:
    flag = False
else:
    flag = True

#if model_name == 'resnet50':
model = custom_models.cam_resnet50()
if flag:
    pretrained = models.resnet50(pretrained=True)
    model = copy_parameters(model, pretrained)
    #removed = list(model.children())[:-2]
    #model = torch.nn.Sequential(*removed)

print(model)

finalconv_name = 'layer4'

model.eval()

model = model.cuda()
# hook the feature extractor


# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

def extract_bbox_from_map(input):
    assert input.ndim == 2, 'Invalid input shape'
    rows = np.any(input, axis=1)
    cols = np.any(input, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def localize_from_map(class_response_map):
    foreground_map = class_response_map > class_response_map.mean()
    objects, count = label(foreground_map)
    max_idx = 0
    max_count = 0
    for obj_idx in range(count):
        count = np.sum(objects==(obj_idx+1))
        if count > max_count:
            max_count = count
            max_idx = obj_idx + 1
    obj = objects==max_idx
    return extract_bbox_from_map(obj)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


model = model.cuda()
root = '/mnt/ramdisk/ImageNet/'
train_imagedir = os.path.join(root, 'train')
val_imagedir = os.path.join(root, 'val')

anno_root = '/opt/caoyh/datasets/ImageNet'
train_annodir = os.path.join(anno_root, 'bbox/train')
val_annodir = os.path.join(anno_root, 'bbox/myval')


classes = os.listdir(val_imagedir)
classes.sort()
result = {}
acc = []
for k, cls in enumerate(classes):
    print('*' * 15)
    print(k, cls)
    total = 0
    IoUSet = []

    files = os.listdir(os.path.join(val_imagedir, cls))
    files.sort()

    for (i, name) in enumerate(files):
        #features_blobs = []
        #model._modules.get(finalconv_name).register_forward_hook(hook_feature)

        xmlfile = os.path.join(val_annodir, cls, name.split('.')[0] + '.xml')
        gt_boxes = get_cls_gt_boxes(xmlfile, cls)

        #raw_img = cv2.imread(os.path.join(val_imagedir, cls, name))
        #h, w, _ = raw_img.shape

        raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
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
        img = preprocess(raw_img)
        img = torch.unsqueeze(img, 0)

        #img = torch.from_numpy(raw_img.transpose(2, 0, 1))
        #img = torch.unsqueeze(img, 0)
        img = to_variable(img)

        logit, feat = model(img)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        #for i in range(0, 5):
        #    print('{:.3f} -> {}'.format(probs[i], idx[i]))

        #CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
        CAMs = returnCAM(feat.data.cpu(), weight_softmax, [k])

        cam_bbox = list(localize_from_map(CAMs[0]))
        #print('cam bbox', cam_bbox) #xmin, ymin, xmax, ymax

        if cam_bbox[0] < 4:
            cam_bbox[0] = 4
        if cam_bbox[1] < 4:
            cam_bbox[1] = 4
        if cam_bbox[2] >220:
            cam_bbox[2] = 220
        if cam_bbox[3] > 220:
            cam_bbox[3] = 220

        #bbox = [cam_bbox[1], cam_bbox[0], cam_bbox[3], cam_bbox[2]]

        #print(gt_boxes)

        max_iou = -1
        for gt_bbox in gt_boxes:
            gt_bbox[0] = int(gt_bbox[0] * ratew)
            gt_bbox[1] = int(gt_bbox[1] * rateh)
            gt_bbox[2] = int(gt_bbox[2] * ratew)
            gt_bbox[3] = int(gt_bbox[3] * rateh)

            iou = IoU(cam_bbox, gt_bbox)

            #print(iou, gt_bbox, cam_bbox)
            if iou > max_iou:
                max_iou = iou

        #print(max_iou)
        #result[os.path.join(cls, name)] = max_iou
        IoUSet.append(max_iou)

        #temp_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        if k<200 and i<1:
            savepath = 'ImageNet/Visualization/CAM_fast/{}/{}'.format(model_name, prefix)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            #print(savepath)
            raw_img = cv2.cvtColor(np.asarray(raw_img), cv2.COLOR_RGB2BGR)
            cv2.rectangle(raw_img, (cam_bbox[0], cam_bbox[1]), (cam_bbox[2], cam_bbox[3]),
                          (0, 255, 255), 4)
            height, width, _ = raw_img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

            #cv2.rectangle(raw_img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]),
            #              (0, 0, 255), 4)

            #cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]), (temp_bbox[2]+temp_bbox[0], temp_bbox[3]+temp_bbox[1]), (0,255,0), 4)
            #for gt_bbox in gt_boxes:
            #    cv2.rectangle(raw_img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]),
            #                  (0, 0, 255), 4)
            result = heatmap * 0.3 + raw_img * 0.7
            cv2.imwrite(os.path.join(savepath, str(name)+'.jpg'), result)

        #'''
    cls_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    print('{} corLoc acc is {}'.format(cls, cls_acc))
    acc.append(cls_acc)

    #break

print(acc)
print(model_name, prefix)
print(np.mean(acc))