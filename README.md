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
