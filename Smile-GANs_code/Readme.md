# Smile-GAN


## Requirement
The code was implemented with the following versions of packages.

- Python 2.7.13
- Pytorch 1.3.1
- Numpy 1.16.6
- Scipy 1.2.3



## Training
For model training, it is required to provide the directory to the data file and the directory where training results will be saved in. The following command can be implemented under the 'models' directory to run ***train.py***

```bash
$ python train.py --data_root <dir1> --save_dir <dir2>
```
Other training options have defaulted values. Detailed explanations can be found in option.py. 

## Testing

After training finished, under the second directory (i.e. \<dir2\>), there are saved checkpoints and one *results.txt* file which records the training process. By loading checkpoints and using functions ***''predict\_cluster''*** and ***''predict\_Y''***, we can obtain probabilities of pattern types for each subject and derive generated patient data along each mapping direction by defining sub variable ***z***.

We have provided one example ***test.py*** file, which can be used directly to check clustering accuracy of trained models on one synthetic test dataset. It is required to provide the directory to the test data file and the directory to the checkpoint. The following command can be implemented under the 'models' directory to run ***test.py***.

```bash
$ python test.py --data_root <dir1> --save_dir <dir2>
```

## Data

Traning the model requires one ***.csv*** file: 

* ***synthetic\_train\_data.csv***: the first and the second column need to be subject ID and diagnosis of subjects, with column name 'ID' and 'diagnosis' respectively. For diagnosis, 0 represents CN and 1 represents patient. Data need to be preprocessed as described in Method Section 4.4. ROI volumes need to be normalized with respect to CN subjects to ensure a mean of 1 and standard deviation of 0.1 among CN subjects for each ROI.


One synthetic train data file ***synthetic\_train\_data.csv*** is provided under directory 'datasets'. These data are generated with simulated disease and non-disease related variations following procedure described in supplementary method section 1.3.1. Datapoints 0:200,200:400,400:600 are selected as subjects with three pre-defined pattern types. 

By setting the option ***--view\_synthetic\_accuracy*** to be ***True***, we can monitor the change of clustering accuracy of each of three subtypes during training process.

Testing also requires one ***.csv*** file:

* ***synthetic\_test\_data.csv***: the first and the second column need to be subject ID and diagnosis of subjects, with column name 'ID' and 'diagnosis' respectively. CN data are no longer required for testing. ROI volumes need to be normalized with respect to CN participants in the training set to ensure a mean of 1 and standard deviation of 0.1 among CN participants for each ROI.

One synthetic test data file ***synthetic\_test\_data.csv*** is also provided under directory 'datasets'. It can be used directly for ***test.py***. However, as clustering is an unsupervised task, one can also check performance of saved models on participants in training set, by simply loading ***synthetic\_train\_data.csv***.

## Pretrained Model
One trained checkpoint with name ***trained_model*** is provided under directory 'training_result'. It can be diretly used for ***test.py*** and should give an clustering accuracy of 1.0.


## Time Requirement
On a 'normal' desktop computer, the training process for the provided synthetic dataset usually converges to the predefined stopping criteria within 4-8 minutes. However, 100% clustering accuracy may not be reached when stopping criteria met (around 95%-100% accuracy). By setting ***--stop*** to be False, the trainig process will continue until the maximum iteration is reached and, each time criteria met, one check point with name ***'converged\_model\_i'*** will be saved. 100% clustering accuracy can be reached in around 8-11 minutes. 



