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

import json
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"
cudnn.benchmark = True

input_size = 224



model = models.resnet50(pretrained=False)
removed = list(model.children())[:-2]

removed.append(nn.MaxPool2d(2,2, padding=1))

model = torch.nn.Sequential(*removed)

model.eval()
model = model.cuda()

root = '/opt/caoyh/datasets/small_imagenet/small_imagenet_total_10000'
imagedir = os.path.join(root, 'train')

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

def get_mask(raw_img):
    img = transform(raw_img)
    img = torch.unsqueeze(img, 0)
    img = to_variable(img)

    vis = model(img)
    vis = to_data(vis)

    #print(vis.size())

    # DDT
    vis = torch.squeeze(vis)
    vis = vis.numpy()

    vis = np.transpose(vis, (1, 2, 0))
    he, we, ce = vis.shape

    vis_sum = np.sum(vis, axis=2)
    #vis_mean = np.mean(vis_sum)
    #vis_mean = np.percentile(vis_sum, 49)

    #highlight = np.zeros((he, we))
    #highlight[vis_sum > vis_mean] = 1

    return vis_sum


def convert_index_to_list(index):
    return list(zip(index[0], index[1]))

def crop_patches(img, FG_index, BG_index, feat_size, patch_size):
    postive_patches = []
    negative_patches = []
    img = np.asarray(img)
    for i in range(feat_size):
        for j in range(feat_size):
            temp = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if (i, j) in FG_index:
                postive_patches.append(temp)
            if (i,j) in BG_index:
                negative_patches.append(temp)

    return postive_patches, negative_patches

def merge_patches(patches_A, patches_B, feat_size, patch_size):
    patches = []
    patches.extend(patches_A)
    patches.extend(patches_B)

    print(len(patches_A), len(patches_B))

    #for i in range(5):
    #    patch = Image.fromarray(patches_A[i])
    #    patch.show()

    m = np.random.permutation(len(patches))
    print(m)

    selected_patches = np.array(patches)[m]

    new_img = np.zeros((feat_size*patch_size, feat_size*patch_size, 3), dtype=np.uint8)

    for i in range(feat_size):
        for j in range(feat_size):
            new_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = selected_patches[i*feat_size+j]

    print(new_img.shape)
    #print(new_img.shape)

    #new_img = Image.new('RGB', (patch_size, patch_size*feat_size*feat_size), 255)

    #for i, patch in enumerate(selected_patches):
    #    new_img.paste(Image.fromarray(patch), (0, patch_size*i))

    #new_img = new_img.resize(feat_size*patch_size, feat_size*patch_size)
    new_img = Image.fromarray(new_img)
    new_img.show()

mask_dict = {}

acc = []
for cls in classes:
    print('*' * 15)
    print(cls)
    total = 0
    IoUSet = []

    files = os.listdir(os.path.join(imagedir, cls))
    files.sort()

    for (i, name) in enumerate(files):
        path = os.path.join(imagedir, cls, name)
        img_A = Image.open(path).convert('RGB')

        img_A = img_A.resize((input_size, input_size))

        mask_A = get_mask(img_A)

        mask_dict[os.path.join(cls, name)] = mask_A.tolist()
    #break
with open('./small_IN_total_10000_mask_4x4.json', 'w') as f:
    json.dump(mask_dict, f)
# load json
with open('./small_IN_total_10000_mask_4x4.json', 'r') as f:
    t = json.load(f)
    keys = list(t.keys())
    mask = np.array(t[keys[0]])
    print(mask.shape)