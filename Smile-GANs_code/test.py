from Smile_GAN_clustering import single_model_clustering, cross_validated_clustering
import pandas as pd

train_data = pd.read_csv('train_roi.csv')
covariate = pd.read_csv('train_cov.csv')

'''
single_model_clustering(train_data, covariate, 4, 0, 10000, '/Users/nyuyzj/Desktop/Smile-GAN/Smile-GAN/Smile-GANs_code/training_result/', 10000, 100000, \
        20, False, saved_model_name='converged_model', random_seed=0, data_fraction=1, lam=9, mu=5, batchSize=25, verbose = True, \
        stop_training = True, beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 5, save_epoch_freq = 5)
'''

cross_validated_clustering(train_data, covariate, 4, 5, 0.8, 100, 200, '/Users/nyuyzj/Desktop/Smile-GAN/Smile-GAN/training_result/', 0.15, 30, \
        0.0015, False, 'highest_matching_clustering', lam=9, mu=5, batchSize=25, verbose = False, \
        beta1 = 0.5, lr = 0.0002, max_gnorm = 100, eval_freq = 10, save_epoch_freq = 5, last_saved_fold = -1)