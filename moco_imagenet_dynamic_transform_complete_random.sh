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
