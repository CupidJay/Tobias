# A Random CNN Sees Objects and Its Applications

Official pytorch implementation of A Random CNN Sees Objects: One Inductive Bias of CNN and Its Applications. This work is accepted by AAAI 2022.

paper is available at [[arxiv]](https://arxiv.org/abs/2106.09259).

## Abstract

This paper starts by revealing a surprising finding: without any learning, a randomly initialized CNN can localize objects surprisingly well. That is, a CNN has an inductive bias to naturally focus on objects, named as Tobias (“<u>T</u>he <u>ob</u>ject <u>i</u>s <u>a</u>t <u>s</u>ight”) in this paper. This empirical inductive bias is further analyzed and successfully applied to self-supervised learning. A CNN is encouraged to learn representations that focus on the foreground object, by transforming every image into various versions with different backgrounds, where the foreground and background separation is guided by Tobias. Experimental results show that the proposed Tobias significantly improves downstream tasks, especially for object detection. This paper also shows that Tobias has consistent improvements on training sets of different sizes, and is more resilient to changes in image augmentations.

## Getting Started

### Prerequisites

* python 3
* PyTorch (= 1.6)
* torchvision (= 0.7)
* Numpy
* CUDA 10.1

### Random CNN's localization ability

For experiments about the localization ability of a random initialized CNN, see [./localization_code](localization_code).



### Tobias SSL——ImageNet experiments

#### Part I: Dataset Preparation

Let's start with small imagenet used in the paper

- First we create small ImageNet, run:

```
python tools/create_small_imagenet.py 
```

- Then we generate localization mask with the help of random CNN for this dataset

```
python tools/generate_mask.py 
```

*Note*: If you want to use other datasets, you just need to change the corresponding lines in generate_mask.py

#### Part II: SSL Pretraining

- For our Tobias SSL, we run moco_imagenet_dynamic_transform.sh:

```
python main_moco_dynamic_transform.py \
  -a resnet50 \
  --lr 0.3 \
  --batch-size 256 --epochs 200 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --moco-k 65536 \
  --bg-prob 0.3 --mask-file ./small_IN_total_10000_mask_4x4.json \
  /opt/caoyh/datasets/small_imagenet/small_imagenet_total_10000
```

*Note:* If you want to change the hyper-parameter p, just use different bg-prob.

- For baseline MoCov2, we run moco_imagenet.sh:

```
python main_moco.py \
  -a resnet50 \
  --lr 0.3 \
  --batch-size 256 --epochs 200 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --moco-k 65536 \
  /opt/caoyh/datasets/small_imagenet/small_imagenet_total_10000
```

- For random merging (kind of pixel-level cutmix), we run moco_imagenet_dynamic_transform_complete_random.sh:

```
python main_moco_dynamic_transform_complete_random.py \
  -a resnet50 \
  --lr 0.3 \
  --batch-size 256 --epochs 800 \
  --input-size 112 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --moco-k 65536 \
  --bg-prob 0.3 --mask-file ./small_IN_total_10000_mask_4x4.json \
  /opt/caoyh/datasets/small_imagenet/small_imagenet_total_10000
```

#### Part III: Downstream Evaluation

**Linear Classification**

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine (c.f. lincls.sh), run:

```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path] \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

**Transferring to Object Detection**

See [./detection](detection).


## Citation

```
@article{Tobias,
   title         = {A Random CNN Sees Objects: One Inductive Bias of CNN and Its Applications},
   author        = {Yun-Hao Cao and Jianxin Wu},
   year          = {2021},
   journal = {arXiv preprint arXiv:2106.09259}}
```