import sys
import os
import numpy as np
from sklearn import metrics
from .model import SmileGAN
from .utils import highest_matching_clustering, consensus_clustering, parse_validation_data
from .clustering import Smile_GAN_train

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


def model_filtering(model_dirs, ncluster, data, covariate=None):
	"""
	Function used for filter out models who have significantly different clustering results with others.
	This function deal with rare failing cases of Smile-GAN
	Args:
		model_dirs: list, list of dirs of all saved models
		ncluster: int, number of defined clusters
		data, data_frame, dataframe with same format as training data. CN data must be exactly same as CN data in training dataframe while 
						  PT data can be any samples in or out of the training set.
		covariate, data_frame, dataframe with same format as training covariate. CN data must be exactly same as CN data in training covariate while 
						  PT data can be any samples in or out of the training set.

	Returns: list of index indicating outlier models

	"""
	_, validation_data = parse_validation_data(data, covariate)
	
	all_prediction_labels = []
	for models in model_dirs:
		model = SmileGAN()
		model.load(models)
		all_prediction_labels.append(np.argmax(model.predict_cluster(validation_data), axis=1))
	model_aris = [[] for _ in range(len(model_dirs))]
	filtered_models = []

	for i in range(len(model_dirs)):
		for j in range(len(model_dirs)):
			if i!=j:
				model_aris[i].append(metrics.adjusted_rand_score(all_prediction_labels[i], all_prediction_labels[j]))
	median_aris = np.median(model_aris, axis=1)
	for j in range(median_aris.shape[0]):
		rest_aris = np.delete(median_aris,j)
		if (median_aris[j]-np.mean(rest_aris))/np.std(rest_aris)<-3:
			filtered_models.append(j)

	return filtered_models

def calculate_ari(prediction_labels):
	model_aris = []
	for i in range(len(prediction_labels)):
		for j in range(i+1,len(prediction_labels)):
			model_aris.append(metrics.adjusted_rand_score(prediction_labels[i], prediction_labels[j]))
	return np.mean(model_aris), np.std(model_aris)



def clustering_result(model_dirs, ncluster, concensus_type, data, covariate=None):
	"""
	Function used for derive clustering results from several saved models
	Args:
		model_dirs: list, list of dirs of all saved models
		ncluster: int, number of defined clusters
		concensus_type: string, the method used for deriving final clustering results with all models derived through CV
								choose between 'highest_matching_clustering' and 'consensus_clustering'
		data, data_frame, dataframe with same format as training data. CN data must be exactly same as CN data in training dataframe while 
						  PT data can be any samples in or out of the training set.
		covariate, data_frame, dataframe with same format as training covariate. CN data must be exactly same as CN data in training covariate while 
						  PT data can be any samples in or out of the training set.
	Returns: clustering outputs.

	"""
	_, validation_data = parse_validation_data(data, covariate)

	all_prediction_labels = []
	all_prediction_probabilities = []
	for models in model_dirs:
		model = SmileGAN()
		model.load(models)
		all_prediction_labels.append(np.argmax(model.predict_cluster(validation_data), axis=1))
		all_prediction_probabilities.append(model.predict_cluster(validation_data))

	if len(model_dirs) > 1:
		mean_ari, std_ari = calculate_ari(all_prediction_labels)
		print("Results have Adjuested_random_index (ARI) = %.2f+- %.2f" %(mean_ari, std_ari))
		if mean_ari<0.3 and concensus_type == 'highest_matching_clustering':
			print('mean ARI < 0.3, concensus_clustering is recommended')
			
	if len(all_prediction_labels) == 1:
		return np.array(all_prediction_labels[0]), np.array(all_prediction_probabilities[0])
	elif concensus_type == 'highest_matching_clustering':
		return highest_matching_clustering(all_prediction_labels, all_prediction_probabilities, ncluster)	
	elif concensus_type == 'consensus_clustering':
		return consensus_clustering(all_prediction_labels, ncluster), None
	else:
		raise Exception("Please choose between 'highest_matching_clustering' and 'consensus_clustering'")



def single_model_clustering(data, ncluster, start_saving_epoch, max_epoch, output_dir, WD_threshold, AQ_threshold, \
		cluster_loss_threshold, covariate=None, saved_model_name='converged_model', lam=9, mu=5, batchSize=25, lipschitz_k = 0.5, verbose = False, \
		beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 5, save_epoch_freq = 5):
	"""
	one of Smile-GAN core function for clustering. Only one model will be trained. (not recommended since result may be not reproducible)
	Args:
		data: dataframe, dataframe file with all ROI (input features) The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be the extracted features. e.g., the ROI features"
		covariate: dataframe, not required; dataframe file with all confounding covariates to be corrected. The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be all confounding covariates. e.g., age, sex"
		ncluster: int, number of defined clusters
		start_saving_epoch: int, epoch number from which model will be saved and training will be stopped if stopping criteria satisfied
		max_epoch: int, maximum trainig epoch: training will stop even if criteria not satisfied.
		output_dir: str, the directory underwhich model and results will be saved
		WD_threshold: int, chosen WD theshold for stopping criteria
		AQ_threshold: int, chosen AQ threhold for stopping criteria
		cluster_loss_threshold: int, chosen cluster_loss threhold for stopping criteria
		load_model: bool, whether load one pre-saved checkpoint
		saved_model_name: str, the name of the saved model
		lam: int, hyperparameter for cluster loss
		mu: int, hyperparameter for change loss
		batchsize: int, batck size for training procedure
		lipschitz_k = float, hyper parameter for weight clipping of mapping and clustering function
		verbose: bool, choose whether to print out training procedure
		beta1: float, parameter of ADAM optimization method
		lr: float, learning rate
		max_gnorm: float, maximum gradient norm for gradient clipping
		eval_freq: int, the frequency at which the model is evaluated during training procedure
		save_epoch_freq: int, the frequency at which the model is saved during training procedure

	Returns: clustering outputs.

	"""

	print('Start Smile-GAN for semi-supervised clustering')

	Smile_GAN_model = Smile_GAN_train(ncluster, start_saving_epoch, max_epoch, WD_threshold, AQ_threshold, \
		cluster_loss_threshold, lam=9, mu=5, batchSize=25, lipschitz_k = 0.5,
		beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 5,save_epoch_freq = 5)
	
	converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, verbose = verbose)
	while not converge:
		print("****** Model not converged at max interation, Start retraining ******")
		converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, verbose = verbose)

	cluster_label, cluster_prob = clustering_result([os.path.join(output_dir,saved_model_name)], ncluster, 'highest_matching_clustering', data, covariate)
	pt_data = data.loc[data['diagnosis'] == 1][['participant_id','diagnosis']]
	pt_data['cluster_label'] = cluster_label + 1

	for i in range(ncluster):
		pt_data['p'+str(i+1)] = cluster_prob[:,i]
	
	pt_data.to_csv(os.path.join(output_dir,'clustering_result.csv'), index = False)

	return pt_data


def cross_validated_clustering(data, ncluster, fold_number, fraction, start_saving_epoch, max_epoch, output_dir, WD_threshold, AQ_threshold, \
		cluster_loss_threshold, concensus_type, covariate=None, lam=9, mu=5, batchSize=25, lipschitz_k = 0.5, verbose = False, \
		beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 5, save_epoch_freq = 5, start_fold = 0, stop_fold = None, check_outlier = True):
	"""
	cross_validated clustering function using Smile-GAN (recommended)
	Args:
		data: dataframe, dataframe file with all ROI (input features) The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be the extracted features. e.g., the ROI features"
		covariate: dataframe, not required; dataframe file with all confounding covariates to be corrected. The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be all confounding covariates. e.g., age, sex"
		ncluster: int, number of defined clusters
		fold_number: int, number of folds for leave-out cross validation
		fraction: float, fraction of data used for training in each fold
		start_saving_epoch: int, epoch number from which model will be saved and training will be stopped if stopping criteria satisfied
		max_epoch: int, maximum trainig epoch: training will stop even if criteria not satisfied.
		output_dir: str, the directory underwhich model and results will be saved
		WD_threshold: int, chosen WD theshold for stopping criteria
		AQ_threshold: int, chosen AQ threhold for stopping criteria
		cluster_loss_threshold: int, chosen cluster_loss threhold for stopping criteria
		###load_model: bool, whether load one pre-saved checkpoint
		concensus_type: string, the method used for deriving final clustering results with all models saved during CV
								choose between 'highest_matching_clustering' and 'consensus_clustering'
		saved_model_name: str, the name of the saved model
		lam: int, hyperparameter for cluster loss
		mu: int, hyperparameter for change loss
		batchsize: int, batck size for training procedure
		lipschitz_k = float, hyper parameter for weight clipping of mapping and clustering function
		verbose: bool, choose whether to print out training procedure
		beta1: float, parameter of ADAM optimization method
		lr: float, learning rate
		max_gnorm: float, maximum gradient norm for gradient clipping
		eval_freq: int, the frequency at which the model is evaluated during training procedure
		save_epoch_freq: int, the frequency at which the model is saved during training procedure
		start_fold; int, indicate the last saved fold index,
							  used for restart previous half-finished cross validation; set defaultly to be 0 indicating a new cv process
		stop_fold: int, indicate the index of fold at which the cv early stop,
							  used for stopping cv process eartly and resuming later; set defaultly to be None and cv will not stop till the end
		check_outlier: bool, whether check outlier model (potential unsuccessful model) after cv process and retrain the fold

	Returns: clustering outputs.

	"""
	
	print('Start Smile-GAN for semi-supervised clustering')

	Smile_GAN_model = Smile_GAN_train(ncluster, start_saving_epoch, max_epoch, WD_threshold, AQ_threshold, \
		cluster_loss_threshold, lam=9, mu=5, batchSize=25, \
		lipschitz_k= 0.5, beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 5,save_epoch_freq = 5)

	saved_models = [os.path.join(output_dir, 'coverged_model_fold'+str(i)) for i in range(fold_number)]
	
	if stop_fold == None:
		stop_fold = fold_number
	for i in range(start_fold, stop_fold):
		print('****** Starting training of Fold '+str(i)+" ******")
		saved_model_name = 'coverged_model_fold'+str(i)
		converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, verbose = verbose)
		while not converge:
			print("****** Model not converged at max interation, Start retraining ******")
			converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, random_seed=i, data_fraction = fraction, verbose = verbose)

	if check_outlier:
		print('****** Start Checking outlier models ******')
		outlier_models = model_filtering(saved_models, ncluster, data, covariate)
		if len(outlier_models) > 0:
			print('Model', end=' ')
			for model in outlier_models:
				print(str(model),end=' ')
			print('have low agreement with other models')
		else:
			print('****** There are no outlier models ******')
	
		for i in outlier_models:
			print('****** Starting training of Fold '+str(i)+" ******")
			saved_model_name = 'coverged_model_fold'+str(i)
			converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, verbose = verbose)
			while not converge:
				print("****** Model not converged at max interation, Start retraining ******")
				converge = Smile_GAN_model.train(saved_model_name, data, covariate, output_dir, verbose = verbose)

	cluster_label, cluster_prob = clustering_result(saved_models, ncluster, concensus_type, data, covariate)
	
	pt_data = data.loc[data['diagnosis'] == 1][['participant_id','diagnosis']]
	pt_data['cluster_label'] = cluster_label + 1

	if concensus_type == "highest_matching_clustering":
		for i in range(ncluster):
			pt_data['p'+str(i+1)] = cluster_prob[:,i]
	
	pt_data.to_csv(os.path.join(output_dir,'clustering_result.csv'), index = False)
	print('****** Smile-GAN clustering finished ******')


if __name__ == "__main__":
	train_model()