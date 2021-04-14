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
python -m torch.distributed.launch --nproc_per_node=NO_GPUS train.py -c config.json
```

`config.json` specifies the configuration to use for training, incl. the path to input images and the values of hyperparameters. the input images must have a `.nii.gz` extension and will be automatically resized to dimensions specified in the configuration file. the directory with the input images must contain subdirectories `segs` with the segmentations and `masks` with the image masks

to test:
```
python -m torch.distributed.launch --nproc_per_node=1 test.py -c config.json -r path/to/checkpoint.pth
```

