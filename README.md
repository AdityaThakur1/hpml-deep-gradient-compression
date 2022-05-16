# Gradient Compression with Delayed Gradient Updates
*Authors:* Aditya Thakur (at4932) , Swapan Jain (sj2594) 

## Overview
- Gradient communication in distributed training accounts for significant bandwidth and limits scalability.
- Empirically it has been found that 99.9% gradient update in Distributed SGD are redundant[1].
- Gradient Compression can reduce latency in communication to a great extent  by compressing gradients.
- Here, we validate the above by  training a ResNet20 Model on a single GPU. Gradients are updated only if they are larger than the threshold.
- We then plot our results of the new model with the baseline ResNet Model.
- We achieve a compression factor of 0.001 with accuracy loss of 3.21%.


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
The code has been adopted from https://github.com/synxlin/deep-gradient-compression


## Training
We use [Horovod](https://github.com/horovod/horovod) to run distributed training:
- run on a machine with 1 GPU without gradient compression (Baseline),
 resnet-20 on cifar-10 dataset with 1 GPU:
```bash
horovodrun -np 1 python train.py --configs configs/cifar/resnet20.py --suffix no_compression --device gpu
```
- run on a machine with 1 GPU with gradient compression ,
 resnet-20 on cifar-10 dataset with 1 GPU:
```bash
horovodrun -np 1 python train.py --configs configs/cifar/resnet20.py configs/dgc/wm5.py configs/dgc/fp16.py configs/dgc/int32.py --suffix with_compression --device gpu
```

## Results

| Model             |  Epochs to converge(85%) | Time to Accuracy (85%) | Test Accuracy (After 200 Epochs) | Train Accuracy (After 200 epochs) | Compression Ratio |  Gradient Update Size |
| ----------------- | ----------- | ----------- |----------- |----------- |----------- |----------- |
| Baseline: ResNet-20 (No Compression)| 78 Epochs  | 11.5 minutes|88.68% | 95.4% | NA |1 MB |
| ResNet-20 With Gradient Compression  | 143 Epochs |  21 minutes | 85.47% |92.5% | 0.001 |1 KB |

## Plots
Below are the plots for Train Accuracy/Loss and Test Accuracy/Loss for Baseline Model and Model with Gradient Compression on ResNet-20.

![Alt text](https://github.com/AdityaThakur1/hpml-deep-gradient-compression/blob/master/data/test_loss_acc.jpeg?raw=true "Test Loss/Accuracy on 200 Epochs")

![Alt text](https://github.com/AdityaThakur1/hpml-deep-gradient-compression/blob/master/data/train_loss_acc.jpeg?raw=true "Train Loss/Accuracy on 200 Epochs")

![Alt text](https://github.com/AdityaThakur1/hpml-deep-gradient-compression/blob/master/data/compression_ratio.jpeg?raw=true "Warmup training and Compression Ratio")
