# Gradient Compression with Delayed Gradient Updates

## Overview

## Content
- [Prerequisites](#prerequisites)
- [Code](#code)
- [Training](#training)

## Prerequisites
The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.5
- [Horovod](https://github.com/horovod/horovod) >= 0.19.4
- [numpy](https://github.com/numpy/numpy)
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [tqdm](https://github.com/tqdm/tqdm)
- [openmpi](https://www.open-mpi.org/software/ompi/) >= 4.0

## Code

The core code to implement Gradient Compression is in [dgc/compression.py](dgc/compression.py) and [dgc/memory.py](dgc/memory.py).

- Gradient Accumulation and Momentum Correction
```python
    mmt = self.momentums[name]
    vec = self.velocities[name]
    if self.nesterov:
        mmt.add_(grad).mul_(self.momentum)
        vec.add_(mmt).add_(grad)
    else:
        mmt.mul_(self.momentum).add_(grad)
        vec.add_(mmt)
    return vec
```

- Sparsification
```python
    importance = tensor.abs()
    # sampling
    sample_start = random.randint(0, sample_stride - 1)
    samples = importance[sample_start::sample_stride]
    # thresholding
    threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
    mask = torch.ge(importance, threshold)
    indices = mask.nonzero().view(-1)
```

## Training
We use [Horovod](https://github.com/horovod/horovod) to run distributed training:
- run on a machine with *N* GPUs,
```bash
horovodrun -np N python train.py --configs [config files]
```
e.g., resnet-20 on cifar-10 dataset with 8 GPUs:
```bash
# fp16 values, int32 indices
# warmup coeff: [0.25, 0.063, 0.015, 0.004, 0.001] -> 0.001
horovodrun -np 8 python train.py --configs configs/cifar/resnet20.py \
    configs/dgc/wm5.py configs/dgc/fp16.py configs/dgc/int32.py
```
- run on *K* machines with *N* GPUs each,
```bash
mpirun -np [K*N] -H server0:N,server1:N,...,serverK:N \
    -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
    python train.py --configs [config files]
```
e.g., resnet-50 on ImageNet dataset with 4 machines with 8 GPUs each,
```bash
# fp32 values, int64 indices, no warmup
mpirun -np 32 -H server0:8,server1:8,server2:8,server3:8 \
    -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
    python train.py --configs configs/imagenet/resnet50.py \
    configs/dgc/wm0.py
```
For more information on horovodrun, please read horovod documentations.

You can modify/add new config files under [configs](configs) to change training settings. You can also modify some trivial configs in the command:
```bash
python train.py --configs [config files] --[config name] [config value] --suffix [suffix of experiment directory]
```
e.g.,
```bash
horovodrun -np 8 python train.py --configs configs/cifar/resnet20.py \
    configs/dgc/wm5.py --configs.train.num_epochs 500 --suffix .e500
```
