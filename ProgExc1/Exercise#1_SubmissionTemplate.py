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


# In[1]:


# import relevant Python libraries

### Your import commands go here ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from IPython.display import display


# # **1. Fetal Health Classification Dataset**

# ## **Clean Dataset**

# ### Problem 1.1

# In[2]:


# load the clean dataset csv file into a pandas dataframe

fetal_df = pd.read_csv('./fetal_health_dataset_clean.csv')


# #### (a)

# The given task is a **Supervised Machine Learning Task**. According to the description of dataset, based on the given features or columns in the dataset we have to classify each observation or row of the dataset into one of the three output classes. The presence of the ***fetal_health*** feature or column which indicates how the given observation or row is classified, makes this dataset ***labelled*** and thus makes this task a **Supervised Machine Learning Task.**

# #### (b)

# In[3]:


print("Axes:-",fetal_df.axes,"Dtypes:-",fetal_df.dtypes,sep="\n\n")


# #### (c)
# 
# 

# In[4]:


display(fetal_df.head(10))


# #### (d)

# In[5]:


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

# In[6]:


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

# In[7]:


fetal_class = fetal_df.groupby(["fetal_health"]).size()
for cls,val in zip(fetal_class.index.values,fetal_class.values) :
    if cls==1 :
        print("Normal: ",val)
    elif cls==2 :
        print("Suspect: ",val)
    else :
        print("Pathological: ",val)


# ## **Dirty Dataset**

# In[14]:


# load the dirty dataset csv file into a pandas dataframe

fetal_dirty_df = pd.read_csv('fetal_health_dataset_dirty.csv')


# ### Problem 1.2

# #### (a)

# In[15]:


# Converts 'NaN' values into 'True' and otherwise into 'False'
matA = fetal_dirty_df.isna().values
print(matA)

print("\n")

# Counts 'False' as 0 and 'True' as 1 and then gives sum of non-zero elements, and thus essentially gives number of NaN values
print("Number of NaN values:",np.count_nonzero(matA))


# #### (b)

# In[16]:


# Converts 'NaN' values into 'True' and otherwise into 'False'
# Counts 'False' as 0 and 'True' as 1 and then gives sum across the columns
print(fetal_dirty_df.isna().sum(axis=0))


# #### (c)

# In[17]:


# Converts 'NaN' values into 'True' and otherwise into 'False'
# Counts 'False' as 0 and 'True' as 1 and then gives sum across the rows
s = fetal_dirty_df.isna().sum(axis=1)
display(s)
print("\n")

# Assign s all those rows where there are 1 or more NaN values
s = s[s > 0]
display(s)
print("\n")
print("Number of samples having missing values:",len(s))


# ### Problem 1.3

# In[18]:


# Printing dataframe info and unique values for all columns
print(fetal_dirty_df.info())
print("\n")
for col in fetal_dirty_df.columns :
    print(col,pd.unique(fetal_dirty_df[col]),sep="\n")
    print("\n\n")

# Plotting box plots for all columns to determine outliers
fig, axes = plt.subplots(22, 1, figsize=(15,22 * 5))
axes = axes.ravel()
for i, col in enumerate(fetal_dirty_df.columns):
    axes[i].boxplot(fetal_dirty_df[col].fillna(0),vert=False)
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Converting negative values in baseline value to NaN
print("Count of NaN values before transformation:",fetal_dirty_df['baseline value'].isna().sum(axis=0))
fetal_dirty_df_bckup = fetal_dirty_df.copy()
fetal_dirty_df['baseline value'] = fetal_dirty_df['baseline value'].map(lambda x: np.nan if x<0 else x,na_action='ignore')
print("Count of NaN values after transformation:",fetal_dirty_df['baseline value'].isna().sum(axis=0))

# Printing rows 91 to 100
display(fetal_dirty_df.loc[91:100])


# After observing the **Dtype**, **Unique Values**, and **Outliers** for all the independent variables, it can be concluded that the variable **"baseline value"** has some kind of logical inconsistency. As per the definition of the feature, it is the FHR value or the number of heart beats per minute for a fetus and thus this value can't be negative, but as we have seen through above analysis that some values in **"baseline value"** variable are negative and thus they are inconsistent.

# ### Problem 1.4

# In[19]:


scaler = StandardScaler()

for col in fetal_dirty_df.columns:
    if col!="fetal_health" and col!="histogram_tendency":
        fetal_dirty_df[col] = scaler.fit_transform(fetal_dirty_df[[col]])

display(fetal_dirty_df.head(20))


# ### Problem 1.5

# In[20]:


resp = "fetal_health"

grouped = fetal_dirty_df.groupby(resp)

for col in fetal_dirty_df.columns:
    if col!=resp:
        fetal_dirty_df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))


# ### Problem 1.6

# In[21]:


enc = OneHotEncoder(handle_unknown='ignore')

fetal_one_hot_df = pd.DataFrame(enc.fit_transform(fetal_dirty_df[["fetal_health"]]).toarray(),columns=["fetal_health_1","fetal_health_2","fetal_health_3"])

fetal_dirty_df = pd.concat([fetal_dirty_df,fetal_one_hot_df],axis=1)

fetal_dirty_df.to_csv("fetal_health_dataset_processed.csv",index=False)


# # **2. Heart Failure Prediction Dataset**

# ## Problem 2.1

# In[22]:


# load the dataset csv file into a pandas dataframe

heart_df = pd.read_csv('./heart_failure_dataset.csv')


# ### (a)

# The given task is a **Supervised Machine Learning Task.** According to the description of dataset, based on the given features or columns in the dataset we have to predict whether a person dies of heart failure or not, which is essentially to classify each observation or row of the dataset into one of the two output classes. The presence of the ***DEATH_EVENT*** feature or column which indicates whether a person dies of heart failure during the follow up period, makes this dataset ***labelled*** and thus makes this task a **Supervised Machine Learning Task.**

# ### (b)

# In[23]:


print("Axes:-",heart_df.axes,"Dtypes:-",heart_df.dtypes,sep="\n\n")


# ### (c)
# 
# 

# In[24]:


heart_df.head(10)


# ### (d)

# In[25]:


print("Rows:",heart_df.shape[0]," Columns:",heart_df.shape[1],sep=" ")


# ### (e)

# Total number of Data Samples : **299**

# ### (f)

# Total number of independent variables : **12.** They are as follows:-
# 1. age
# 2. anaemia
# 3. creatinine_phosphokinase
# 4. diabetes
# 5. ejection_fraction
# 6. high_blood_pressure
# 7. platelets
# 8. serum_creatinine
# 9. serum_sodium
# 10. sex
# 11. smoking
# 12. time

# ### (g)

# Total number of dependent variables: **1**. It is **DEATH_EVENT**.

# ### (h)

# n = **299** and p = **12**

# ### (i)

# n = **299** and m = **1**

# ### (j)

# In[26]:


for col in heart_df.columns :
    print(col,pd.unique(heart_df[col]),sep="\n")
    print("\n\n")
    
fig, axes = plt.subplots(7, 2, figsize=(15,7 * 5))
axes = axes.ravel()
for i, col in enumerate(heart_df.columns):
    axes[i].hist(heart_df[col], bins=20, color='blue', alpha=0.7) 
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# From above unique values and histograms, it can be observed that only few variables show some kind of discrete nature ranging over only 0 or 1 and so can be encoded into categorical values. These variables are mainly -- **anaemia, diabeters, high_blood_pressue, sex, smoking** and **DEATH_EVENT**. Also referring the dataset description and logical definition of variables it can be inferred as well that these are the only variables which can be properly categorized as classes. The above variables takes only **binary values** which is another indication that they can be encoded. Therefore, this dataset contains **6** categorical variables.

# ### (k)

# **Nominal Encoding**

# ### (l)

# In[27]:


death_class = heart_df.groupby(["DEATH_EVENT"]).size()
for cls,val in zip(death_class.index.values,death_class.values) :
    if cls==0 :
        print("Remaining Paitients: ",val)
    else:
        print("Deceased Patients: ",val)


# ### (m)

# In[28]:


sex_class = heart_df.groupby(["sex"]).size()
for cls,val in zip(sex_class.index.values,sex_class.values) :
    if cls==0 :
        print("Women: ",val)
    else:
        print("Men: ",val)


# ### (n)

# In[29]:


smoking_class = heart_df.groupby(["smoking"]).size()
for cls,val in zip(smoking_class.index.values,smoking_class.values) :
    if cls==0 :
        print("Non-smokers: ",val)
    else:
        print("Smokers: ",val)


# ## Problem 2.2

# In[30]:


# Printing dataframe info and unique values for all columns
print(heart_df.info())
print("\n")
for col in heart_df.columns :
    print(col,pd.unique(heart_df[col]),sep="\n")
    print("\n\n")

# Plotting box plots for all columns to determine outliers
fig, axes = plt.subplots(13, 1, figsize=(15,13 * 5))
axes = axes.ravel()
for i, col in enumerate(heart_df.columns):
    axes[i].boxplot(heart_df[col].fillna(0),vert=False)
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

print("Count of NaN values before transformation:",heart_df['age'].isna().sum(axis=0))
heart_df_bckup = heart_df.copy()
heart_df['age'] = heart_df['age'].map(lambda x: x if x==int(x) else np.nan,na_action='ignore')
print("Count of NaN values after transformation:",heart_df['age'].isna().sum(axis=0))


# After observing the **Dtype**, **Unique Values**, and **Outliers** for all the features, it can be concluded that the variable **"age"** has some kind of logical inconsistency. As per the definition of the feature, it is the age of a person in years and can be only an integer, but as we have seen through above analysis that some values in **"age"** variable are fractional and thus they are inconsistent.

# ## Problem 2.3

# In[31]:


scaler = StandardScaler()
cat_cols = set(["anaemia", "diabetes", "high_blood_pressue", "sex", "smoking","DEATH_EVENT"])

for col in heart_df.columns:
    if col not in cat_cols:
        heart_df[col] = scaler.fit_transform(heart_df[[col]])

display(heart_df.head(20))


# ## Problem 2.4

# In[32]:


enc = OneHotEncoder(handle_unknown='ignore')

heart_one_hot_df = pd.DataFrame(enc.fit_transform(heart_df[["DEATH_EVENT"]]).toarray(),columns=["DEATH_EVENT_0","DEATH_EVENT_1"])

heart_df = pd.concat([heart_df,heart_one_hot_df],axis=1)

heart_df.to_csv("heart_failure_dataset_processed.csv",index=False)


# ## Problem 2.5

# In[33]:


display(heart_df.corr())


# ### (a)

# The two most positively coorelated variables with **DEATH_EVENT** are **serum_creatinine** and **age**.

# ### (b)

# The two most negatively coorelated variables with **DEATH_EVENT** are **time** and **ejection_fraction**.

# ### (c)

# The second most positively correlated variable with **DEATH_EVENT** is **age**. This correlation is consistent with the fact that people with higher age have a higher chance or risk of heart failure than people with lower age, and thus people with high age have higher probability of DEATH_EVENT being 1.

# ### (d)

# **time** : As per the definition, it is the number of follow-up days to visit a doctor and if a person dies then it is the number of follow-up days before a person died during the follow-up period, and so higher the value of this variable is, lower is the chance of that person dying as they are regularly visiting for follow-up and thus are alive. So the **time** variable is negatively correlated with the **DEATH_EVENT**, higher the value of **time** makes **DEATH_EVENT** tends to 0 and vice versa.
# 
# **ejection_fraction**: As per the definition, it is the percentage of blood leaving the heart after each contraction. This implies that the larger this value is, higher is the chance that the heart is not functioning properly as large amount of blood loss from heart is an indicator of some underlying issue and thus the probability of that person dying increases. So the **ejection_fraction** variable is negatively correlated with the **DEATH_EVENT**, higher the value of **ejection_fraction** makes **DEATH_EVENT** tends to 0 and vice versa.
