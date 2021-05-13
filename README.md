# Smile-GAN
Smile-GAN is a semi-supervised clustering method which is designed to identify disease-related heterogeneity among the patient group. The model effectively avoid variations among normal control (CN) group and cluster patient based on disease related variations only. Semi-supervised clustering of Smile-GAN is achieved through joint training of the mapping and clustering function, where the mapping function can map CN subjects along different mapping directions depending on disease-related variations.

![image info](./datasets/Smile-GAN.png)

## License
Copyright (c) 2016 University of Pennsylvania. All rights reserved. See[ https://www.cbica.upenn.edu/sbia/software/license.html](https://www.cbica.upenn.edu/sbia/software/license.html)

## Installation
We highly recommend the users to install **Anaconda3** on your machine. After installing Anaconda3, Smile-GAN can be used following this procedure:

We recommend the users to use the Conda virtual environment:

```bash
$ conda create --name smilegan python=3.8
```
Activate the virtual environment

```bash
$ conda activate smilegan
```

Install other SmileGAN package dependencies (go to the root folder of SmileGAN)

```bash
$ ./install_requirements.sh
```

Finally, we need to install SmileGAN from PyPi:

```bash
$ pip install SmileGAN
```



## Input structure
Main functions of SmileGAN basically takes two panda dataframes as data inputs, **data** and **covariate** (optional). Columns with name *'participant_id'* and *diagnosis* must exist in both dataframes. Some conventions for the group label/diagnosis: -1 represents healthy control (CN) and 1 represents patient (PT); categorical variables, such as sex, should be encoded to numbers: Female for 0 and Male for 1, for example. 

Example for **data**:

```bash
participant_id    diagnosis    ROI1    ROI2 ...
subject-1		    -1         325.4   603.4
subject-2 		     1         260.5   580.3
subject-3           -1         326.5   623.4
subject-4            1         301.7   590.5
subject-5            1		   293.1   595.1
subject-6            1   	   287.8   608.9
```
Example for **covariate**

```bash
participant_id    diagnosis    age    sex ...
subject-1		    -1         57.3   0
subject-2 		     1         43.5   1
subject-3           -1         53.8   1
subject-4            1         56.0   0
subject-5            1		   60.0   1
subject-6            1   	   62.5   0
```

## Example
We offer a toy dataset in the folder of SmileGAN/dataset.

**Runing SmileGAN for clustering CN vs Subtype1 vs Subtype2 vs ...**

```bash
import pandas as pd
from SmileGAN.Smile_GAN_clustering import single_model_clustering,cross_validated_clustering

train_data = pd.read_csv('train_roi.csv')
covariate = pd.read_csv('train_cov.csv')

output_dir = "PATH_OUTPUT_DIR"
ncluster = 3
start_saving_epoch = 6000
max_epoch = 10000

## three parameters for stopping threhold
WD = 0.11
AQ = 20
clustering_error = 0.0015

## clustering without cross validation (train model only once, not recommended)
single_model_clustering(train_data, ncluster, start_saving_epoch, max_epoch,\
					    output_dir, WD, AQ, clustering_error, covariate=covariate)
					    
## clustering with cross validation (recommended)
fold_number = 10  # number of folds the leave-out cv runs
data_fraction = 0.8 # fraction of data used in each fold
cross_validated_clustering(train_data, ncluster, start_saving_epoch, max_epoch,\
					    output_dir, WD, AQ, clustering_error, covariate=covariate)
```

When using the package, ***WD***, ***AQ***, ***clustering\_error*** need to be chosen empirically. WD is recommended to be a value between 0.1-0.15, AQ is recommended to be 1/20 of patient numbers and clustering_error is recommended to be a value between 0.0015-0.002. A large value of ***start\_saving\_epoch*** can better guarantee the convergence of the model though requires longer training time.

Since the CV process may take long training time on a normal desktop computer, the function **cross\_validated\_clustering** enables early stop and later resumption. Users can set ***stop\_fold*** to be early stopping point and ***start_fold*** depending on previous stopping point.


## Citation
If you use this package for research, please cite the following papers:

```bash
@misc{yang2020smilegans,
      title={Smile-GANs: Semi-supervised clustering via GANs for dissecting brain disease heterogeneity from medical images}, 
      author={Zhijian Yang and Junhao Wen and Christos Davatzikos},
      year={2020},
      eprint={2006.15255},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

```bash
@misc{yang2021BrainHeterogeneity,
      title={Disentangling brain heterogeneity via semi-supervised deep-learning and MRI: dimensional representations of Alzheimer's Disease}, 
      author={Zhijian Yang and Ilya M. Nasrallah and Haochang Shou and Junhao Wen and Jimit Doshi and Mohamad Habes and Guray Erus and Ahmed Abdulkadir and Susan M. Resnick and David Wolk and Christos Davatzikos},
      year={2021},
      eprint={2102.12582},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


