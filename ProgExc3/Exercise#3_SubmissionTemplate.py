#!/usr/bin/env python
# coding: utf-8

# # [Amet Vikram]
# # Programming Exercise \#3
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


# # **1. Noise Level Classification Using Plug-in Classifiers**

# ## 1.1

# ### (a)

# In[3]:


df = pd.read_csv("NoiseClassificationTrainingData.csv")

# Applying groupby on df based on "ClassLabel"
groups = df.groupby("ClassLabel")

# Getting only "NoiseLevel" column for further processing
grouped = groups["NoiseLevel"]

# print("Total number of classes:",len(grouped.groups))
# print("\n")
# print("The label name of all classes:", grouped.groups.keys())
# print("\n")

# nj vector which stores count of individual classes
n_j = grouped.count()

# n_total scaler value which stores the total count sum of all classes
n_total = n_j.sum()

# nj_sum vector which stores the individual sum of all classes
nj_sum = grouped.sum()

# mean_j vector which stores the individual mean of all classes
mean_j = nj_sum / n_j
print("For 'ClassLabel' 0, 1, 2 :-")
print("The individual mean of all classes in same order are :",mean_j.values)
print("\n")

# Now here we will apply formula for common covariance matrix but
# since we have only one feature vector the equation will reduce to 
# calculating simple scaler common variance. Although we will use variable M
# as we have unknown means in the equation for each class.
total_sum = 0
M = 3
for key in grouped.groups.keys():
    s = np.square(grouped.get_group(key) - mean_j[key])
    total_sum += s.sum()
common_variance = total_sum / (n_total - M)

print("The common variance for three classes is:",common_variance)


# ### (b)

# _[Your answer for 1.1(b) goes here]_

# ## 1.2

# ### (a)

# In[ ]:





# ### (b)

# _[Your answer for 1.2(b) goes here]_

# ## 1.3

# In[4]:


# Calculates prior probabilities based on the count of individual classes
prior_prob = n_j / n_total
print("For 'ClassLabel' 0, 1, 2 :-")
print("The individual prior probabilities of all classes in same order are :",prior_prob.values)


# ## 1.4

# In[8]:


# Calculates value of gaussian probability function for given 
# value of x , given mean and given variance
def plugin_gaussian_function(x, mean, variance):
    p = (np.square(x - mean) / variance) * (-0.5)
    num = np.power(np.e, p)
    denum = np.sqrt(2*np.pi*variance)
    return (num / denum)

# Calcualtes ML decision class using ML estimate 
# mean and common variance
def compute_using_ML_classifier(x):
    res = []
    for key in grouped.groups.keys():
        res.append(plugin_gaussian_function(x, mean_j[key], common_variance))
    return np.array(res).argmax()

# Calculates MAP decision class using ML estimated
# mean,prior probability and common variance
def compute_using_MAP_classifier(x):
    res = []
    for key in grouped.groups.keys():
        res.append(plugin_gaussian_function(x, mean_j[key], common_variance) * prior_prob[key])
    return np.array(res).argmax()

# Calculates General Loss Bayes decision classfier for 
# given loss matrix and using ML estimated mean, prior
# probability and common variance
#
# NOTE: Posterior Probability vector: A is multiplied
# by Loss Matrix: L using matrix multiplication such that
# resultant vector: B gives the Posterior probability adjusted 
# by missclassification loss.
# AL = B
L = np.array([[-1, 2, 4], [2, 0, 4], [4, 4, 0]])
def compute_using_GeneralLossBayes_classifier(x):
    res = []
    for key in grouped.groups.keys():
        res.append(plugin_gaussian_function(x, mean_j[key], common_variance) * prior_prob[key])
    A = np.array(res).reshape(1,3)
    B = np.dot(A,L)
    return B.argmin()

df_test = pd.read_csv("NoiseClassificationTestData.csv")

res_ml = df_test["NoiseLevel"].apply(compute_using_ML_classifier)

res_map = df_test["NoiseLevel"].apply(compute_using_MAP_classifier)

res_glb = df_test["NoiseLevel"].apply(compute_using_GeneralLossBayes_classifier)

print("For ML classifier:-")
print("Number of samples classified as class '0':", len(res_ml[res_ml==0]))
print("Number of samples classified as class '1':", len(res_ml[res_ml==1]))
print("Number of samples classified as class '2':", len(res_ml[res_ml==2]))
print("Number of samples correctly classified:", (res_ml==df_test["ClassLabel"]).sum())
print("\n")
print("For MAP classifier:-")
print("Number of samples classified as class '0':", len(res_map[res_map==0]))
print("Number of samples classified as class '1':", len(res_map[res_map==1]))
print("Number of samples classified as class '2':", len(res_map[res_map==2]))
print("Number of samples correctly classified:", (res_map==df_test["ClassLabel"]).sum())
print("\n")
print("For General Loss Bayes classifier:-")
print("Number of samples classified as class '0':", len(res_glb[res_glb==0]))
print("Number of samples classified as class '1':", len(res_glb[res_glb==1]))
print("Number of samples classified as class '2':", len(res_glb[res_glb==2]))
print("Number of samples correctly classified:", (res_glb==df_test["ClassLabel"]).sum())


# ## 1.5

# ### (a)

# In[ ]:


### Your code for 1.5(a) goes here ###


# ### (b)

# In[ ]:


### Your code for 1.5(b) goes here ###


# ## 1.6

# ### (a)

# _[Your answer for 1.6(a) goes here]_

# ### (b)

# _[Your answer for 1.6(b) goes here]_

# ### (c)

# _[Your answer for 1.6(c) goes here]_

# ### (d)

# _[Your answer for 1.6(d) goes here]_
