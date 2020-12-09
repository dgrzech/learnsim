image registration via SG-MCMC
============
code for the paper "Image registration via stochastic gradient Markov chain Monte Carlo"


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
to register a pair of images:
```
python run.py -vi 1 -mcmc 1 -d device_id -c config.json
```

`config.json` specifies the configuration to use for training, incl. the path to input images and the values of hyperparameters. the input images must have a `.nii.gz` extension and will be automatically resized to dimensions specified in the configuration file. the directory with the input images must contain subdirectories `seg` with the segmentations and `masks` with the image masks

to resume:
```
python train.py -r path/to/last/checkpoint.pth
```
