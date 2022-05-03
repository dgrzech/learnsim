## A variational Bayesian method for similarity learning in non-rigid image registration (CVPR 2022)
[Daniel Grzech](https://www.linkedin.com/in/dgrzech/)<sup>1</sup>, [Mohammad Farid Azampour](https://www.linkedin.com/in/mfazampour/)<sup>1,2,3</sup>, [Ben Glocker](https://www.imperial.ac.uk/people/b.glocker)<sup>1</sup>, [Julia Schnabel](https://scholar.google.co.uk/citations?user=FPykfZ0AAAAJ&hl)<sup>3,4</sup>, [Nassir Navab](https://campar.in.tum.de/Main/NassirNavabCv)<sup>3,5</sup>, [Bernhard Kainz](http://bernhard-kainz.com/)<sup>1,6</sup>, and [Lo&iuml;c Le Folgoc](https://scholar.google.co.uk/citations?user=_7MXR2MAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup> Imperial College London, <sup>2</sup> Sharif University of Technology, <sup>3</sup> Technische Universit&auml;t M&uuml;nchen, <sup>4</sup> King's College London, <sup>5</sup> Johns Hopkins University, <sup>6</sup> Friedrich-Alexander-Universit&auml;t Erlangen-N&uuml;rnberg

### Pre-print 
[<img src="figures/thumbnail.png" width="75%" height="75%">](preprint.pdf)

### Abstract

We propose a novel variational Bayesian formulation for diffeomorphic non-rigid registration of medical images, which learns in an unsupervised way a data-specific similarity metric. The proposed framework is general and may be used together with many existing image registration models. We evaluate it on brain MRI scans from the UK Biobank and show that use of the learnt similarity metric, which is parametrised as a neural network, leads to more accurate results than use of traditional functions to which we initialise the model, e.g. SSD and LCC, without a negative impact on image registration speed or transformation smoothness. In addition, the method estimates the uncertainty associated with the transformation.

### Results

<img src="figures/result.png" width="75%" height="75%">

The output on two sample images in the test split when using the baseline and the learnt similarity metrics. In case of SSD, the average improvement in Dice scores over the baseline on the image above is approximately 27.2 percentage points and in case of LCC, it is approximately 6.$ percentage points. The uncertainty estimates are visualised as the standard deviation of the displacement field, based on 50 samples. Use of the learnt similarity metric which was initialised to SSD also results in better calibration of uncertainty estimates than in case of the baseline.

<img src="figures/result_VXM.png" width="90%" height="90%">

The output on two sample images in the test split when using VoxelMorph trained with the baseline and the learnt similarity metrics. In order to make the comparison fair, we use the exact same hyperparameter values for VoxelMorph trained with the baseline and the learnt similarity metrics. In case of VoxelMorph + SSD, the average improvement in Dice scores over the baseline on the image above is approximately 25.3 percentage points and in case of LCC, it is approximately 11.8 percentage points.

<img src="figures/boxplot.png" width="75%" height="75%">

Average surface distances and Dice scores calculated on subcortical structure segmentations when aligning images in the test split using the baseline and learnt similarity metrics. The learnt models show clear improvement over the baselines. We provide details on the statistical significance of the improvement in the paper.

### Code
[GitHub](https://github.com/dgrzech/learnsim)

### Citation

Daniel Grzech, Mohammad Farid Azampour, Ben Glocker, Julia Schnabel, Nassir Navab, Bernhard Kainz, and Loïc Le Folgoc. <b>A variational Bayesian method for similarity learning in non-rigid image registration</b>. CVPR 2022.

Click [here](Grzech2022.bib) for a .bib file.

[<img src="figures/logos/Imperial.png" width="25%" height="25%">](https://biomedia.doc.ic.ac.uk/) &nbsp; &nbsp; [<img src="figures/logos/Sharif.png" width="10%" height="10%">](https://en.sharif.edu/) &nbsp; &nbsp; [<img src="figures/logos/TUM.png" width="25%" height="25%">](https://www.tum.de/)

[<img src="figures/logos/Kings.png" width="15%" height="15%">](https://www.kcl.ac.uk/) &nbsp; &nbsp; [<img src="figures/logos/JHU.png" width="20%" height="20%">](hhttps://www.jhu.edu/) &nbsp; &nbsp; [<img src="figures/logos/FAU.png" width="10%" height="10%">](https://www.fau.eu/)

[<img src="figures/logos/CDT.png" width="15%" height="15%">](https://www.imagingcdt.com/) &nbsp; &nbsp; [<img src="figures/logos/UKRI.png" width="15%" height="15%">](https://www.ukri.org/)