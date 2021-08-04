LearnSim
============

usage
------------
to train the similarity metric:
```
CUDA_VISIBLE_DEVICES=<device_ids> python -m torch.distributed.launch --nproc_per_node=NO_GPUS train.py -c path/to/config.json
```

`config.json` specifies the configuration to use for training, incl. the similarity metric parametrisation and the values of hyperparameters

to test:
```
CUDA_VISIBLE_DEVICES=<device_id> python -m torch.distributed.launch --nproc_per_node=1 test.py -c path/to/config.json -r path/to/checkpoint.pt
```
