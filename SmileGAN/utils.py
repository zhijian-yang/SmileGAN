import numpy as np
import scipy
import pandas as pd
import itertools
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from .data_loading import PTIterator, CNIterator, val_PT_construction, val_CN_construction

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

def find_highest_ari_model(clustering_results):
    """
    Find one of save models which have the highest average overlap (evaluated by ARI) with all other saved models
    The returned model will be used as a template such that all other models will \
    be reordered to achieve the highest overlap with it.
    :param clustering_results: list, list of clustering results given by all saved models; 
                               length of list equals the number of saved models.
    :return: int, the index of selected model with highest average overlap
    """
    highest_ari=0
    best_model=0
    for i in range(len(clustering_results)):
        all_ari=[]
        for j in range(len(clustering_results)):
            if i!=j:all_ari.append(metrics.adjusted_rand_score(clustering_results[i], clustering_results[j]))
        if np.mean(all_ari)>highest_ari:
            best_model=i
            highest_ari=np.mean(all_ari)
    return best_model


def get_model_order(cluster_results,ncluster):
    """
    Find best orders for results given by all saved models so that they reach highest agreements
    :param clustering_results: list, list of clustering results given by all saved models; 
                               length of list equals the number of saved models.
    :param ncluster: int, number of clusters
    :return: list, list of best orders for results given by all saved models.
    """
    order_permutation = list(itertools.permutations(range(ncluster)))
    best_model = find_highest_ari_model(cluster_results)
    all_orders=[]
    for k in range(len(cluster_results)):
        if k==best_model: 
            all_orders.append(range(ncluster))
        elif k!=best_model:
            highest_intersection = 0
            for order in order_permutation:
                total_intersection = 0
                for i in range(ncluster):
                    total_intersection+= np.intersect1d(np.where(cluster_results[best_model]==i),np.where(cluster_results[k]==order[i])).shape[0]
                if total_intersection>=highest_intersection:
                    best_order=order
                    highest_intersection=total_intersection
            all_orders.append(best_order)
    return all_orders

def highest_matching_clustering(clustering_results, label_probability, ncluster):
    """
    The function which offers clustering result (cluster label and cluster probabilities) 
    by reordering and combining clustering results given by all saved models
    :param clustering_results: list, list of clustering results given by all saved models; 
                               length of list equals the number of saved models.
    :param label_probability: list, list of clutering results given by all saved modesl;
                                length of list equals the number of saved models and each 
                                model gives an n*k list.
    :param ncluster: int, number of clusters
    :return: two arrays, one n*1 array with cluster label for each participant
                         one n*k array with k cluster probabilites for each participant
    """
    order = get_model_order(clustering_results, ncluster)
    class_index=0
    for i in range(len(clustering_results)):
        label_probability[i] = label_probability[i][:,order[i]]
    prediction_prob=np.mean(label_probability,axis=0)
    prediction_cluster=prediction_prob.argmax(axis=1)
    return prediction_cluster, prediction_prob, 

def Covariate_correction(cn_data,cn_cov,pt_data,pt_cov):
    """
    Eliminate the confound of covariate, such as age and sex, from the disease-based changes.
    :param cn_data: array, control data
    :param cn_cov: array, control covariates
    :param pt_data: array, patient data
    :param pt_cov: array, patient covariates
    :return: corrected control data & corrected patient data
    """
    pt_cov = (pt_cov-np.amin(cn_cov, axis=0))/(np.amax(cn_cov, axis=0)-np.amin(cn_cov, axis=0))
    cn_cov = (cn_cov-np.amin(cn_cov, axis=0))/(np.amax(cn_cov, axis=0)-np.amin(cn_cov, axis=0))
    beta = np.transpose(LinearRegression().fit(cn_cov, cn_data).coef_)
    corrected_cn_data = cn_data-np.dot(cn_cov,beta)
    corrected_pt_data = pt_data-np.dot(pt_cov,beta)
    return corrected_cn_data, corrected_pt_data

def Data_normalization(cn_data,pt_data):
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1 
    among CN participants for each ROI
    :param cn_data: array, control data
    :param pt_data: array, patient data
    :return: normalized control data & normalized patient data
    """
    cn_mean = np.mean(cn_data,axis=0)
    cn_std = np.std(cn_data,axis=0)
    normalized_cn_data = 1+(cn_data-cn_mean)/(10*cn_std)
    normalized_pt_data = 1+(pt_data-cn_mean)/(10*cn_std)
    return normalized_cn_data, normalized_pt_data

def parse_train_data(data, covariate, random_seed, data_fraction, batch_size):
    cn_data = data.loc[data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values
    pt_data = data.loc[data['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
    if covariate is not None:
        cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values
        pt_cov = covariate.loc[covariate['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
        cn_data,pt_data = Covariate_correction(cn_data,cn_cov,pt_data,pt_cov)
    normalized_cn_data, normalized_pt_data = Data_normalization(cn_data,pt_data)
    cn_train_dataset = CNIterator(normalized_cn_data, random_seed, data_fraction, batch_size)
    pt_train_dataset = PTIterator(normalized_pt_data, random_seed, data_fraction, batch_size)
    return cn_train_dataset, pt_train_dataset


def parse_validation_data(data, covariate):
    cn_data = data.loc[data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values
    pt_data = data.loc[data['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
    if covariate is not None:
        cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values
        pt_cov = covariate.loc[covariate['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
        cn_data,pt_data = Covariate_correction(cn_data,cn_cov,pt_data,pt_cov)
    normalized_cn_data, normalized_pt_data = Data_normalization(cn_data,pt_data)
    cn_eval_dataset = val_CN_construction(normalized_cn_data).load()
    pt_eval_dataset = val_PT_construction(normalized_pt_data).load()
    return cn_eval_dataset, pt_eval_dataset



def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the numpy array is symmetric or not
    """
    result = np.allclose(a, a.T, rtol=rtol, atol=atol)
    return result

def consensus_clustering(clustering_results, k):
    """
    This function performs consensus clustering on a co-occurence matrix
    :param clustering_results: an array containing all the clustering results given by all saved models
    :param k: number of clusters
    :return:
    """
    clustering_results = np.transpose(np.array(clustering_results))
    num_pt = clustering_results.shape[0]
    cooccurence_matrix = np.zeros((num_pt, num_pt))

    for i in range(num_pt - 1):
        for j in range(i + 1, num_pt):
            cooccurence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

    cooccurence_matrix = np.add(cooccurence_matrix, cooccurence_matrix.transpose())
    ## here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(cooccurence_matrix, axis=1)), cooccurence_matrix)

    Laplacian_norm = np.subtract(np.eye(num_pt), np.matmul(np.matmul(np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1))), cooccurence_matrix), np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1)))))
    ## replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    ## check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    if check_symmetric(Laplacian_norm):
        ## extract the eigen value and vector
        ## matlab eig equivalence is eigh, not eig from numpy or scipy, see this post: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        ## Note, the eigenvector is not unique, thus the matlab and python eigenvector may be different, but this will not affect the results.
        evalue, evector = scipy.linalg.eigh(Laplacian_norm)
    else:
        # evalue, evector = np.linalg.eig(Laplacian_norm)
        raise Exception("The Laplacian matrix should be symmetric here...")

    ## check if the eigen vector is complex
    if np.any(np.iscomplex(evector)):
        evalue, evector = scipy.linalg.eigh(Laplacian)

    ## create the kmean algorithm with sklearn
    kmeans = KMeans(n_clusters=k, n_init=20).fit(evector.real[:, 0: k])
    final_predict = kmeans.labels_

    return final_predict



