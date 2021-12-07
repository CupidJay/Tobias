import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



def load_pretrained_model(model, pretrained):
    #loading from mocov2 pretrained models
    if os.path.isfile(pretrained):
        print("=> loading pretrained from checkpoint {}".format(pretrained))

        checkpoint = torch.load(pretrained, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        #state_dict = checkpoint
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('module.') and not k.startswith('module.fc'):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)
        #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(pretrained))
    #loading from ImageNet pretrained models
    elif pretrained in model_names:
        print("=> loading pretrained from ImageNet pretrained {}".format(pretrained))
        checkpoint = models.__dict__[pretrained](pretrained=True)
        state_dict = checkpoint.state_dict()
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pretrained from ImageNet pretrained {}".format(pretrained))
    else:
        print("=> NOT load pretrained")
    return model

def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item==0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d:d[1], reverse=True)
    return count_dict[0][0]


def sk_pca(X, k):
    from sklearn.decomposition import PCA
    pca = PCA(k)
    pca.fit(X)
    vec = pca.components_
    #print(vec.shape)
    return vec

def fld(x1, x2):
    x1, x2 = np.mat(x1), np.mat(x2)
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    k = x1.shape[1]

    m1 = np.mean(x1, axis=0)
    m2 = np.mean(x2, axis=0)
    m = np.mean(np.concatenate((x1, x2), axis=0), axis=0)
    print(x1.shape, m1.shape)


    c1 = np.cov(x1.T)
    s1 = c1*(n1-1)
    c2 = np.cov(x2.T)
    s2 = c2*(n2-1)
    Sw = s1/n1 + s2/n2
    print(Sw.shape)
    W = np.dot(np.linalg.inv(Sw), (m1-m2).T)
    print(W.shape)
    W = W / np.linalg.norm(W, 2)
    return np.mean(np.dot(x1, W)), np.mean(np.dot(x2, W)), W

def pca(X, k):
    n, m = X.shape
    mean = np.mean(X, 0)
    #print(mean.shape)
    temp = X - mean
    conv = np.cov(X.T)
    #print(conv.shape)
    conv1 = np.cov(temp.T)
    #print(conv-conv1)

    w, v = np.linalg.eig(conv)
    #print(w.shape)
    #print(v.shape)
    index = np.argsort(-w)
    vec = np.matrix(v.T[index[:k]])
    #print(vec.shape)

    recon = (temp * vec.T)*vec+mean

    #print(X-recon)
    return vec


def get_bbox_dict(root):
    print('loading from ground truth bbox')
    name_idx_dict = {}
    with open(os.path.join(root, 'images.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, name = fileline[0], fileline[1]
            name_idx_dict[name] = idx

    idx_bbox_dict = {}
    with open(os.path.join(root, 'bounding_boxes.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, bbox = fileline[0], list(map(float, fileline[1:]))
            idx_bbox_dict[idx] = bbox

    name_bbox_dict = {}
    for name in name_idx_dict.keys():
        name_bbox_dict[name] = idx_bbox_dict[name_idx_dict[name]]

    return name_bbox_dict





