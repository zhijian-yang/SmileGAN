import pandas as pd
from SmileGAN.Smile_GAN_clustering import cross_validated_clustering, single_model_clustering
import os

if __name__ == '__main__':
	output_dir = './smile'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	train_data = pd.read_csv('sample_data.csv')

	ncluster = 3
	start_saving_epoch = 6000
	max_epoch = 14000
	WD = 0.14
	AQ = 30
	cluster_loss = 0.003


	cross_validated_clustering(train_data, ncluster, 3, 0.8, start_saving_epoch, max_epoch, output_dir, WD, AQ, cluster_loss,\
		'highest_matching_clustering', batchSize=25, lipschitz_k=0.5)



