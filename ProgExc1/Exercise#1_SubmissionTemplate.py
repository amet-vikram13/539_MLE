#!/usr/bin/env python
# coding: utf-8

# # [Amet Vikram]
# # Programming Exercise \#1
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


# In[]:


# import relevant Python libraries

### Your import commands go here ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # **1. Fetal Health Classification Dataset**

# ## **Clean Dataset**

# ### Problem 1.1

# In[3]:


# load the clean dataset csv file into a pandas dataframe

fetal_df = pd.read_csv('./fetal_health_dataset_clean.csv')


# #### (a)

# The given task is a **Supervised Machine Learning Task**. According to the description of dataset, based on the given features or columns in the dataset we have to classify each observation or row of the dataset into one of the three output classes. The presence of the ***fetal_health*** feature or column which indicates how the given observation or row is classified, makes this dataset ***labelled*** and thus makes this task a **Supervised Machine Learning Task.**

# #### (b)

# In[4]:


print("Axes:-",fetal_df.axes,"Dtypes:-",fetal_df.dtypes,sep="\n\n")


# #### (c)
#
#

# In[5]:


fetal_df.head(10)


# #### (d)

# In[6]:


print("Rows:",fetal_df.shape[0]," Columns:",fetal_df.shape[1],sep=" ")


# #### (e)

# Within machine learning paralance, each row can be termed as a ***Data Sample*** or ***Observation***.

# #### (f)

# Total number of Data Samples : **2126**

# #### (g)

# Total number of independent variables : **21**. They are as follows:-
# 1. baseline value
# 2. accelerations
# 3. fetal_movement
# 4. uterine_contractions
# 5. light_decelerations
# 6. severe_decelerations
# 7. prolongued_decelerations
# 8. abnormal_short_term_variability
# 9. mean_value_of_short_term_variability
# 10. percentage_of_time_with_abnormal_long_term_variability
# 11. mean_value_of_long_term_variability
# 12. histogram_width
# 13. histogram_min
# 14. histogram_max
# 15. histogram_number_of_peaks
# 16. histogram_number_of_zeroes
# 17. histogram_mode
# 18. histogram_mean
# 19. histogram_median
# 20. histogram_variance
# 21. histogram_tendency

# #### (h)

# Total number of dependent variables : **1**. It is ***fetal_health.***

# #### (i)

# n = **2126** and p = **21**.

# #### (j)

# n = **2126** and m = **1**.

# #### (k)

# Based on the dataset, the data seems to be preprocessed. There is an indication that data in some column have been transformed. For instance, the ***fetal_health*** variable or feature is supposed to be one of three types -- Normal, Suspect and Pathological, but in the dataset it seems to be encoded, specifically **Ordinal Encoding**. Similarly the feature or variable ***histogram_tendency*** which by definition represents the shape of histogram and can be one of negatively skewed, symmetric, or positively skewed, also seems to be encoded. Additionally, there are no data samples with any missing data values, which suggests that the dataset does not require ***imputing missing values***. All these reasons suggest that the dataset has gone through some kind of transformation.

# #### (l)

# In[18]:


for col in fetal_df.columns :
    print(col,pd.unique(fetal_df[col]),sep="\n")
    print("\n\n")

fig, axes = plt.subplots(11, 2, figsize=(15,11 * 5))
axes = axes.ravel()
for i, col in enumerate(fetal_df.columns):
    axes[i].hist(fetal_df[col], bins=20, color='blue', alpha=0.7)
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# From above unique values and histograms, it can be observed that only few variables show some kind of discrete nature which can be encoded into categorical values. These variables are mainly -- ***histogram_tendency, fetal_health, histogram_number_of_zeros, prolonged_decelerations,*** and ***severe_decelerations***, but based on the dataset description and logical definition of variables it can be inferred that only ***fetal_health*** and ***histogram_tendency*** variables are the only variables which can be properly categorized as classes. ***fetal_health*** feature, as mentioned in dataset description, can be one of three classes and ***histogram_tendency*** which by definition represents shape of histogram can be again categorized in classes. Therefore, this dataset only contain **2** categorical variables.

# #### (m)

# **Ordinal Encoding**

# #### (n)

# In[99]:


fetal_class = fetal_df.groupby(["fetal_health"]).size()
for cls,val in zip(fetal_class.index.values,fetal_class.values) :
    if cls==1 :
        print("Normal: ",val)
    elif cls==2 :
        print("Suspect: ",val)
    else :
        print("Pathological: ",val)


# ## **Dirty Dataset**

# In[ ]:


# load the dirty dataset csv file into a pandas dataframe

### fetal_dirty_df = pd.read_csv('fetal_health_dataset_dirty.csv') ###


# ### Problem 1.2

# #### (a)

# In[ ]:


### Your code for 1.2(a) goes here ###


# #### (b)

# In[ ]:


### Your code for 1.2(b) goes here ###


# #### (c)

# In[ ]:


### Your code for 1.2(c) goes here ###


# ### Problem 1.3

# In[ ]:


### Your code for 1.3 goes here ###


# _[Your justification for 1.3 goes here]_

# ### Problem 1.4

# In[ ]:


### Your code for 1.4 goes here ###


# ### Problem 1.5

# In[ ]:


### Your code for 1.5 goes here ###


# ### Problem 1.6

# In[ ]:


### Your code for 1.6 goes here ###


# # **2. Heart Failure Prediction Dataset**

# ## Problem 2.1

# In[ ]:


# load the dataset csv file into a pandas dataframe

### heart_df = pd.read_csv('heart_failure_dataset.csv') ###


# ### (a)

# _[Your answer for 2.1(a) goes here]_

# ### (b)

# In[ ]:


### Your code for 2.1(b) goes here ###


# ### (c)
#
#

# In[ ]:


### Your code for 2.1(c) goes here ###


# ### (d)

# In[ ]:


### Your code for 2.1(d) goes here ###


# ### (e)

# _[Your answer for 2.1(e) goes here]_

# ### (f)

# _[Your answer for 2.1(f) goes here]_

# ### (g)

# _[Your answer for 2.1(g) goes here]_

# ### (h)

# _[Your answer for 2.1(h) goes here]_

# ### (i)

# _[Your answer for 2.1(i) goes here]_

# ### (j)

# In[ ]:


### Your code for 2.1(j) goes here ###


# _[Your justification for 2.1(j) goes here]_

# ### (k)

# _[Your answer for 2.1(k) goes here]_

# ### (l)

# In[ ]:


### Your code for 2.1(l) goes here ###


# ### (m)

# In[ ]:


### Your code for 2.1(m) goes here ###


# ### (n)

# In[ ]:


### Your code for 2.1(n) goes here ###


# ## Problem 2.2

# In[ ]:


### Your code for 2.2 goes here ###


# _[Your justification for 2.2 goes here]_

# ## Problem 2.3

# In[ ]:


### Your code for 2.3 goes here ###


# ## Problem 2.4

# In[ ]:


### Your code for 2.4 goes here ###


# ## Problem 2.5

# In[ ]:


### Your code for 2.5 goes here ###


# ### (a)

# _[Your answer for 2.5(a) goes here]_

# ### (b)

# _[Your answer for 2.5(b) goes here]_

# ### (c)

# _[Your answer for 2.5(c) goes here]_

# ### (d)

# _[Your answer for 2.5(d) goes here]_
