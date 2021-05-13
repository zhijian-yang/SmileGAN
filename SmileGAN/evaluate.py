import numpy as np
from scipy.linalg import sqrtm
from .model import sample_z

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

## calulate wasserstein distance assuming independent ROIs
def cal_w_distance(mean_1,mean_2,std_1,std_2):
    distance=0
    for i in range(mean_1.size):
        distance+=((mean_1[i]-mean_2[i])**2+(std_1[i])**2+std_2[i]**2-2*std_1[i]*std_2[i])
    return  distance
    
## calulate wasserstein distance without assuming independent ROIs
def cal_w_distance_with_cov(mean_1,mean_2,std_1,std_2):
    distance=0
    distance=np.sum(np.square(mean_1-mean_2))+np.trace(std_1+std_2-2*sqrtm(sqrtm(std_2).dot(std_1).dot(sqrtm(std_2))))
    return  distance.real

## return mean wasserstein distance and wasserstein distance of each mapping direction
def cal_validate_distance(model,mean,std,eva_data,independent):
    predicted_z=model.predict_cluster(eva_data)
    cluster=[[]for k in range(model.opt.ncluster)]
    distance=[1000 for m in range(model.opt.ncluster)]
    adjusted_eva_data=eva_data.detach().numpy()
    for i in range(predicted_z.shape[0]):
        a=predicted_z[i].tolist().index(max(predicted_z[i].tolist()))
        cluster[a].append(adjusted_eva_data[i])
    for j in range(model.opt.ncluster):
        if len(cluster[j])>1:
            cluster_mean=np.mean(np.array(cluster[j]), axis=0)
            if independent:
                cluster_std=np.std(np.array(cluster[j]), axis=0)
                distance[j]=cal_w_distance(mean[j],cluster_mean,std[j],cluster_std)
            else:
                cluster_std=np.cov(np.transpose(np.array(cluster[j])))
                distance[j]=cal_w_distance_with_cov(mean[j],cluster_mean,std[j],cluster_std)
    output=[0 for _ in range(model.opt.ncluster)]
    for k in range(model.opt.ncluster):
        if distance[k]!=1000:
            output[k] = distance[k]
    return np.mean(output),output
      
def eval_w_distances(real_X,real_Y,model,independent):
    predict_Y=[]
    mean=[]
    std=[]
    for i in range(model.opt.ncluster):
        z, z_index = sample_z(real_X, model.opt.ncluster, fix_class=i)
        predict_Y.append(model.predict_Y(real_X,z).detach().numpy())
    for j in range(model.opt.ncluster):
        mean.append(np.mean(predict_Y[j], axis=0))
        std.append(np.std(predict_Y[j], axis=0)) if independent else std.append(np.cov(np.transpose(np.array(predict_Y[j]))))
    max_distance, w_distances=cal_validate_distance(model,mean,std,real_Y,independent)
    return max_distance, w_distances

def multiple_cluster_accuracy(prediction):
    output=[]
    for sub in prediction:
        sum=0
        accuracy_list=[0 for _ in range(len(prediction))]
        for index in range(len(prediction)):
            for i in range(sub.shape[0]):
                if (sub[i].tolist().index(max(sub[i].tolist()))==index): sum+=1
            accuracy_list[index]=sum*1.0/sub.shape[0]
            sum=0
        output.append(accuracy_list)
    return output

## return clustering distribution of subjects in three predefined subtypes
def cluster_output(model,test_Y,opt):
    prediction=model.predict_cluster(test_Y)
    return multiple_cluster_accuracy([prediction[0:200],prediction[200:400],prediction[400:]])

## return clustering memberships of valY datapoints and quantity of subjects assigned to each subtype
def label_change(model,test_Y,opt):
    predicted_label=[]
    predicted_class=[0 for _ in range(opt.ncluster)]
    prediction=model.predict_cluster(test_Y)
    for i in range(prediction.shape[0]):
        label=prediction[i].tolist().index(max(prediction[i].tolist()))
        predicted_label.append(label)
        predicted_class[label]+=1
    return predicted_label, predicted_class






