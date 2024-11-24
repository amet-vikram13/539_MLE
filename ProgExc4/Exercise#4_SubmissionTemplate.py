#!/usr/bin/env python
# coding: utf-8

# # [Amet Vikram]
# # Programming Exercise \#4
# ---

# # Preamble

# In[ ]:


# optional code cell when using Google Colab with Google Drive

# remove the docstring comment block below in order to mount Google Drive
'''
# mount Google Drive in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# change directory using the magic command %cd
### replace [MY PATH] below with your own path in Google Drive ###
### %cd /content/drive/My\ Drive/[MY PATH] ###
'''


# In[1]:


# import relevant Python libraries

import numpy as np
import pandas as pd
from scipy import stats as sps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Latex


# # **1. Machine Learning for Diagnosis of 'Senioritis'**

# ## Problem 1.1

# In[2]:


df_train = pd.read_csv("SenioritisTrainingData.csv")
df_test = pd.read_csv("SenioritisTestData.csv")

groupby_train = df_train.groupby("ClassLabel")
groupby_test = df_test.groupby("ClassLabel")

fig, ax = plt.subplots(2, 1, figsize=(12,12))

class_labels = groupby_train.groups.keys()

ax[0].set_title("Training Data")
ax[0].set_xlabel("ChemA")
ax[0].set_xlabel("ChemB")
for label in class_labels:
    x = groupby_train.get_group(label)["ChemA"]
    y = groupby_train.get_group(label)["ChemB"]
    color = "#2ca02c" if label=="No Senioritis" else "#ff0000"
    ax[0].scatter(x,y,c=color)
ax[0].legend(class_labels)

ax[1].set_title("Test Data")
ax[1].set_xlabel("ChemA")
ax[1].set_ylabel("ChemB")
for label in class_labels:
    x = groupby_test.get_group(label)["ChemA"]
    y = groupby_test.get_group(label)["ChemB"]
    color = "#2ca02c" if label=="No Senioritis" else "#ff0000"
    ax[1].scatter(x,y,c=color)
ax[1].legend(class_labels)

plt.tight_layout()
plt.show()


# ## Problem 1.2

# ### (a)

# In[17]:


## Now for LDA, we know as it is a plugin MAP classifier
## with different means but same covariance matrix, where
## the likelihood functions are Gaussian, and so we need
## to estimate the following parameters :-
## 1) prior probabilities
## 2) mean vector
## 3) common covariance matrix

## Ensuring feature order and label order as :-
## ChemA : 0 ; ChemB : 1
## No Senioritis: 0 ; Senioritis : 1
## for all subsequent calculations

##
## input: pandas.DataFrame
## input: class label column name
## output: pandas.Series
##
def estimate_prior_probability(data,column_name):
    label_order = ["No Senioritis","Senioritis"]
    n_total = len(data)
    gr = data.groupby(column_name)[column_name]
    n_j = gr.count()[label_order]
    return n_j / n_total

print("The Prior Probabilities are:-")
print(estimate_prior_probability(df_train,"ClassLabel"))
print("\n")

##
## input: pandas.DataFrame
## input: class label column name
## output: pandas.DataFrame
##
def estimate_mean_vectors(data,column_name):
    feature_order = ["ChemA","ChemB"]
    gr = data.groupby(column_name)
    return gr.mean()[feature_order]

print("The Mean of classes are:-")
print(estimate_mean_vectors(df_train,"ClassLabel"))
print("\n")

##
## input: pandas.DataFrame
## input: class label column name
## output: numpy.ndarray
##
def estimate_common_covariance_matrix(data,column_name):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    mean_matrix = estimate_mean_vectors(data,column_name)
    
    gr = data.groupby(column_name)

    ## Initializing shape of common covariance matrix
    itr_sum = np.zeros((2,2))

    n_total = len(data)
    M = len(label_order)
    
    for label in label_order:
        class_data = gr.get_group(label)[feature_order]
        mean_vec = mean_matrix.loc[label][feature_order].to_numpy().reshape(2,1)
        
        for i in range(len(class_data)):
            curr_vec = class_data.iloc[i].to_numpy().reshape(2,1)
            temp_sum = np.dot((curr_vec - mean_vec),(curr_vec - mean_vec).transpose())
            itr_sum += temp_sum

    return itr_sum / (n_total - M)

print("The Common Covariance Matrix is:-")
print(estimate_common_covariance_matrix(df_train,"ClassLabel"))
print("\n")


## LDA Classifier
## δj(x) := arg j max [ (xT @ inv(C) @ mj) + (-1/2)*(mjT @ inv(C) @ mj) + ln(πj) ] NOTE: @ -> dot Product
## Term1 := xT @ inv(C) @ mj
## Term2 := (-1/2)*(mjT @ inv(C) @ mj)
## Term3 := ln(πj)
##
## input: pandas.DataFrame
## input: class label column name
## input: pandas.DataFrame
## output: pandas.Series
##
def classifier_LDA(data_train,column_name,data_test):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    C = estimate_common_covariance_matrix(data_train,column_name)
    mean_matrix = estimate_mean_vectors(data_train,column_name)
    prior_prob = estimate_prior_probability(data_train,column_name)

    C_inv = np.linalg.inv(C)

    res = []
    for i in range(len(data_test)):
        y_hat = []
        x  = data_test[feature_order].iloc[i].to_numpy().reshape(2,1)
        
        for label in label_order:
            mj = mean_matrix.loc[label][feature_order].to_numpy().reshape(2,1)
            πj = prior_prob.loc[label]
            
            term1 = np.linalg.multi_dot([x.transpose(), C_inv, mj])
            term2 = (-0.5)*np.linalg.multi_dot([mj.transpose(), C_inv, mj])
            term3 = np.log(πj)
            
            y_hat.append(term1 + term2 + term3)
            
        y_hat = np.array(y_hat)
        res.append(label_order[y_hat.argmax()])

    return pd.Series(res,name="ClassLabel_LDA")


# ### (b)

# In[18]:


## Now for QDA, we know as it is a plugin MAP classifier
## with different means and different covariance matrices, where
## the likelihood functions are Gaussian, and so we need
## to estimate the following parameters :-
## 1) prior probabilities
## 2) mean vector
## 3) covariance matrices

## Ensuring feature order and label order as :-
## ChemA : 0 ; ChemB : 1
## No Senioritis: 0 ; Senioritis : 1
## for all subsequent calculations

## From previous part, we already have functions
## to estimate, prior probabilities and mean vectors

##
## input: pandas.DataFrame
## input: class label column name
## output: python list
##
def estimate_covariance_matrices(data,column_name):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    mean_matrix = estimate_mean_vectors(data,column_name)
    
    gr = data.groupby(column_name)
    
    ## Initializing shape of covariance matrix
    itr_sum = np.zeros((2,2))
    
    n_total = len(data)
    
    res = []
    for label in label_order:
        class_data = gr.get_group(label)[feature_order]
        mean_vec = mean_matrix.loc[label][feature_order].to_numpy().reshape(2,1)
        nj = gr[column_name].count().loc[label]
        
        for i in range(len(class_data)):
            curr_vec = class_data.iloc[i].to_numpy().reshape(2,1)
            temp_sum = np.dot((curr_vec - mean_vec),(curr_vec - mean_vec).transpose())
            itr_sum += temp_sum
            
        res.append(itr_sum / (nj - 1))

    return pd.Series(res,index=label_order)

print("The Covariance Matrices are:-")
for mat in estimate_covariance_matrices(df_train,"ClassLabel"):
    print(mat)
    print("\n")

## QDA Classifier
## δj(x) := arg j max [ (-1/2)*(xT @ inv(Cj) @ x) + (xT @ inv(Cj) @ mj) + (-1/2)*(mjT @ inv(Cj) @ mj) + (-1/2)*ln(Det(Cj)) + ln(πj) ] NOTE: @ -> dot Product
## Term1 := (-1/2)*(xT @ inv(Cj) @ x)
## Term2 := xT @ inv(C) @ mj
## Term3 := (-1/2)*(mjT @ inv(C) @ mj)
## Term4 := (-1/2)*ln(Det(Cj))
## Term5 := ln(πj)
##
## input: pandas.DataFrame
## input: class label column name
## input: pandas.DataFrame
## output: pandas.Series
##
def classifier_QDA(data_train,column_name,data_test):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    mean_matrix = estimate_mean_vectors(data_train,column_name)
    prior_prob = estimate_prior_probability(data_train,column_name)

    Cj = estimate_covariance_matrices(data_train,column_name)

    res = []
    for i in range(len(data_test)):
        y_hat = []
        x  = data_test[feature_order].iloc[i].to_numpy().reshape(2,1)
        
        for label in label_order:
            mj = mean_matrix.loc[label][feature_order].to_numpy().reshape(2,1)
            πj = prior_prob.loc[label]
            Cj_inv = np.linalg.inv(Cj.loc[label])
            
            term1 = (-0.5)*np.linalg.multi_dot([x.transpose(), Cj_inv, x])
            term2 = np.linalg.multi_dot([x.transpose(), Cj_inv, mj])
            term3 = (-0.5)*np.linalg.multi_dot([mj.transpose(), Cj_inv, mj])
            term4 = (-0.5)*np.log(np.linalg.det(Cj.loc[label]))
            term5 = np.log(πj)
            
            y_hat.append(term1 + term2 + term3 + term4 + term5)
            
        y_hat = np.array(y_hat)
        res.append(label_order[y_hat.argmax()])

    return pd.Series(res,name="ClassLabel_QDA")


# ### (c)

# In[19]:


## Now for Gaussian Naive Bayes, we know as it is a plugin MAP classifier
## where the factorization of the joint likelihood in terms of the marginal 
## likelihoods leads to product of indvidual gaussian pmf for each 
## feature and so we need to estimate the following parameters for each
## feature :-
## 1) prior probabilities
## 2) mean
## 3) variance

## Ensuring feature order and label order as :-
## ChemA : 0 ; ChemB : 1
## No Senioritis: 0 ; Senioritis : 1
## for all subsequent calculations

## From previous part, we already have functions
## to estimate, prior probabilities and means

##
## input: pandas.DataFrame
## input: class label column name
## output: pandas.DataFrame
##
def estimate_var_vectors(data,column_name):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    mean_matrix = estimate_mean_vectors(data,column_name)
    
    gr = data.groupby(column_name)

    res = []
    for label in label_order:
        va = []
        nj = gr[column_name].count().loc[label]
        for feat in feature_order:
            class_data = gr.get_group(label)[feat]
            va.append(np.sum(np.square(class_data - mean_matrix.loc[label,feat])) / (nj - 1))
        res.append(va)

    return pd.DataFrame(res,index=label_order,columns=feature_order)

print("The Variance of classes are:-")
print(estimate_var_vectors(df_train,"ClassLabel"))
print("\n")

# Calculates value of gaussian probability function for given 
# value of x , given mean and given variance
def plugin_gaussian_function(x, mean, variance):
    p = (np.square(x - mean) / variance) * (-0.5)
    num = np.power(np.e, p)
    denum = np.sqrt(2*np.pi*variance)
    return (num / denum)

## GNB Classifier
## δj(x) := arg j max [ πj * Product_of_all_plugin_gaussian(Pxk(x / Y=j)) ]
##
## input: pandas.DataFrame
## input: class label column name
## input: pandas.DataFrame
## output: pandas.Series
##
def classifier_GNB(data_train,column_name,data_test):
    label_order = ["No Senioritis","Senioritis"]
    feature_order = ["ChemA","ChemB"]
    
    mean_matrix = estimate_mean_vectors(data_train,column_name)
    var_matrix = estimate_var_vectors(data_train,column_name)
    prior_prob = estimate_prior_probability(data_train,column_name)

    res = []
    for i in range(len(data_test)):
        y_hat = []
        x = data_test.iloc[i]
        
        for label in label_order:
            πj = prior_prob.loc[label]
            inst = 1
            
            for feat in feature_order:
                mk = mean_matrix.loc[label,feat]
                vk = var_matrix.loc[label,feat]
                inst *= plugin_gaussian_function(x.loc[feat],mk,vk)
                
            y_hat.append(πj*inst)

        y_hat = np.array(y_hat)
        res.append(label_order[y_hat.argmax()])

    return pd.Series(res,name="ClassLabel_GNB")    


# ### (d)

# In[20]:


## Now for KNN classifier, we know that for a given
## test data point x, it is classified to that label
## for which, label is in "majority" in the "neighbourhood" of
## point x0, where x0 is the point closest to x.

## Ensuring feature order and label order as :-
## ChemA : 0 ; ChemB : 1
## No Senioritis: 0 ; Senioritis : 1
## for all subsequent calculations

## maintains a list of tuple(distance_metrix, index),
## of length K in sorted order, pops the last elements
## to maintain length K.
def insertion_sort(dist_list,d):
    idx = -1
    for i in range(len(dist_list)-1,-1,-1):
        val = dist_list[i][0]
        if d[0] > val:
            idx = i
            break
    dist_list.insert(idx+1,d)

    dist_list.pop()
    return dist_list

## for a given neighbourhood i.e the K closes points
## computes the label for that neighbourhood
def find_majority_neighbourhood(data,dist_list):
    class_0 = 0
    class_1 = 0
    for _,i in dist_list:
        class_0 += 1 if "No Senioritis"==data.iloc[i].loc["ClassLabel"] else 0
        class_1 += 1 if "Senioritis"==data.iloc[i].loc["ClassLabel"] else 0
    ans = "No Senioritis" if class_0 > class_1 else "Senioritis"
    return ans

## KNN Classifier
def classifier_KNN(data,K):
    assert K<len(data)
    
    feature_order = ["ChemA","ChemB"]

    res_labels = []

    ## maintains a "upper triangular matrix"
    ## of distances computed between index i and j
    full_nbh = []
    for i in range(len(data)):
        itr = []
        for j in range(len(data)):
            itr.append(False)
        full_nbh.append(itr)
    
    for i in range(0,len(data)):
        nbh = [(np.inf,-1) for i in range(K)]
        
        xi = data.iloc[i].loc[feature_order].to_numpy()

        ## checking from previously computed distances
        ## as L2(i,j) == L2(j,i)
        for j in range(0,i):
            val = full_nbh[j][i]
            if val!=False:
                nbh = insertion_sort(nbh, (val,j))

        for j in range(i+1,len(data)):
            x = data.iloc[j].loc[feature_order].to_numpy()
            d = np.linalg.norm(x - xi,ord=2)
            nbh = insertion_sort(nbh, (d,j))
            full_nbh[i][j] = d
        
        res_labels.append(find_majority_neighbourhood(data,nbh))
    
    return pd.Series(res_labels,name="ClassLabel_KNN")


# ## Problem 1.3

# ### (a)

# In[7]:


### Your code for 1.3(a) goes here ###


# ### (b)

# In[8]:


### Your code for 1.3(b) goes here ###


# ### (c)

# In[9]:


### Your code for 1.3(c) goes here ###


# ### (d)

# In[10]:


### Your code for 1.3(d) goes here ###


# ### (e)

# In[11]:


### Your code for 1.3(e) goes here ###


# ## Problem 1.4

# _[Your answer for 1.4 goes here]_

# ## Problem 1.5

# ### (a)

# In[12]:


### Your code for 1.5(a) goes here ###


# ### (b)

# In[13]:


### Your code for 1.5(b) goes here ###


# ### (c)

# In[14]:


### Your code for 1.5(c) goes here ###


# ### (d)

# In[15]:


### Your code for 1.5(d) goes here ###


# ### (e)

# In[16]:


### Your code for 1.5(e) goes here ###


# ## Problem 1.6
# 

# ### (a)

# In[17]:


### Your code for 1.6(a) goes here ###


# ### (b)

# _[Your answer for 1.6(b) goes here]_
