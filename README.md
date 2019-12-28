LearnSim
============
a variational Bayesian method for similarity learning in medical image registration

training
------------
to start training:
```
python train.py -c config.json
```

`config.json` specifies the configuration to use for training, incl. the path to input images and the type of similarity metric to use at initialisation (SSD or LCC). the input images must have a `nii.gz` extension and will be automatically downsampled to dimensions of 128^3 voxels. by default, registration will be carried out in an all-to-one manner.

to resume training:
```
python train.py -r path/to/last/checkpoint.pth
```
