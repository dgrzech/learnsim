LearnSim
============


dependencies
------------
* NiBabel
* matplotlib
* numpy
* pandas
* PyTorch
* scikit-learn
* SimpleITK
* tvtk


usage
------------
to train the similarity metric:
```
python -m torch.distributed.launch --nproc_per_node=NO_GPUS train.py -c path/to/config.json
```

`config.json` specifies the configuration to use for training, incl. the similarity metric parametrisation and the values of hyperparameters

to test:
```
python -m torch.distributed.launch --nproc_per_node=1 test.py -c path/to/config.json -r path/to/checkpoint.pth
```
