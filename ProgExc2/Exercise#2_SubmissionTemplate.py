#!/usr/bin/env python
# coding: utf-8

# # [Amet Vikram]
# # Programming Exercise \#2
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


# # **1. Feature Engineering for _Environmental Sensor Telemetry Data_**

# ## Problem 1.1

# In[2]:


df = pd.read_csv("./iot_telemetry_dataset.csv")


# ### (a)

# The given task can be treated as a **supervised learning task.** Depending on our choice, any one of the **Sensor's telemetry variable** that is, **Temperature, humidity, CO, liquid petroleum gas (LPG), smoke, light, and motion** can act as a **response variable** and remainig variables can act as **regressor or predictor variables.** For instance one can choose to predict, the humidity in environment based on the other variables' data. It should be noted that the following task can also be treated as an **Exploratory Data Analysis** task, where one can simply choose to analyze relationship between different telemetry variables' in an environment over a time frame.

# ### (b)

# In[3]:


print(df.info())
display(df.head())
print(f"Total number of data samples : {df.shape[0]}")


# ### (c)
# 
# 

# In[4]:


mac_addr = "00:0f:00:70:91:0a"
print("Number of data samples associated with device {} : {}".format(mac_addr,len(df[df["device"]==mac_addr])))


# 
# ### (d)

# In[5]:


mac_addr = "1c:bf:ce:15:ec:4d"
print("Number of data samples associated with device {} : {}".format(mac_addr,len(df[df["device"]==mac_addr])))


# ### (e)

# In[6]:


mac_addr = "b8:27:eb:bf:9d:51"
print("Number of data samples associated with device {} : {}".format(mac_addr,len(df[df["device"]==mac_addr])))


# ## Problem 1.2

# ### (a)

# In[7]:


numerical_cols = ["co","lpg","smoke","temp", "humidity"]
group_col = "device"

groups = df.groupby(group_col)

grouped = groups[numerical_cols]

means = grouped.mean()
variances = grouped.var()

print(means)
print("\n")
print(variances)
print("\n")

# Here for visualization purpose we will seperate out our variables in two 
# groups as not all variables are on same scale, and thus will not be visible
# clearly on a bar plot.
# Group 1 : "co","lpg","smoke"
# Group 2 : "temp", "humidity"
means_group_one = means[numerical_cols[:3]]
means_group_two = means[numerical_cols[3:]]
variances_group_one = variances[numerical_cols[:3]]
variances_group_two = variances[numerical_cols[3:]]

means_group_one.plot(kind='bar', figsize=(10, 6))
plt.title('Means of Variables in Group 1 by Device')
plt.ylabel('Mean Value')
plt.xlabel('Device')
plt.legend(title='Variable')
plt.grid(True)
plt.show()

means_group_two.plot(kind='bar', figsize=(10, 6))
plt.title('Means of Variables in Group 2 by Device')
plt.ylabel('Mean Value')
plt.xlabel('Device')
plt.legend(title='Variable')
plt.grid(True)
plt.show()

variances_group_one.plot(kind='bar', figsize=(10, 6))
plt.title('Variances of Variables in Group 1 by Device')
plt.ylabel('Variance')
plt.xlabel('Device')
plt.legend(title='Variable')
plt.grid(True)
plt.show()

variances_group_two.plot(kind='bar', figsize=(10, 6))
plt.title('Variances of Variables in Group 2 by Device')
plt.ylabel('Variance')
plt.xlabel('Device')
plt.legend(title='Variable')
plt.grid(True)
plt.show()


# Based on the above plots, there are few observations to note :-
# 1) **co** and **lpg** have around same means and variance for all three devices.
# 2) **smoke** has a very high mean and variance compared to **co** and **lpg** for all three devices.
# 3) **smoke** for device 1 (left most) has a very high variance compared to variance of other two devices.
# 4) **temp** variance for device 2 (middle one) is very high compared to other two devices.

# ### (b)

# In[8]:


def standardization(arr):
    mean = np.mean(arr)
    std  = np.std(arr)
    return (arr - mean)/(std)

for col in numerical_cols:
    df[col] = groups[col].transform(standardization)


# 
# ### (c)

# In[9]:


categorical_cols = ["device","light","motion"]
df = pd.get_dummies(df,columns=categorical_cols)


# ### (d)

# In[10]:


df.head(20)


# ### (e)

# The **ts** variable as by definition from the given data set is the timestamp at a particular time when the sensor records data. It is currently in numerical format but can be converted into appropriate timestamp format. It can be used to analyze the variables transformation over a period of time.

# ## Problem 1.3

# In[11]:


co_cidx = -1
humidity_cidx = -1
lpg_cidx = -1
smoke_cidx = -1
temp_cidx = -1

for i,col in enumerate(df.columns):
    if col=="co":
        co_cidx = i
    if col=="humidity":
        humidity_cidx = i
    if col=="lpg":
        lpg_cidx = i
    if col=="smoke":
        smoke_cidx = i
    if col=="temp":
        temp_cidx = i

def compute_stats_features(row):
    f_mean  = np.mean(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]]))
    f_gmean = sps.gmean(np.abs(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]])))
    f_hmean = sps.hmean(np.abs(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]])))
    f_var   = np.var(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]]))
    f_kurt  = sps.kurtosis(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]]))
    f_skew  = sps.skew(np.array([row[co_cidx],row[humidity_cidx],row[lpg_cidx],row[smoke_cidx],row[temp_cidx]]))
    return np.array([f_mean,f_gmean,f_hmean,f_var,f_kurt,f_skew])
    
stats_df = pd.DataFrame(np.apply_along_axis(compute_stats_features, axis=1, arr=df),columns=["mean","gmean","hmean","variance","kurtosis","skewness"])

display(stats_df.head(40))


# # **2. Empirical Risk Minimization and the Law of Large Numbers**

# ## Problem 2.1

# ### (a)

# In[12]:


die_outcomes = [1,2,3,4,5,6]
n_samples = 10000

sims = np.random.choice(die_outcomes,size=n_samples)

# Calculates cumulative sum i.e calculates Di for all i in (1,n_samples)
sims_cumsum = np.cumsum(sims)

# Creates array sample size n for Di for all i in (1,n_samples)
n_arr = np.arange(1,n_samples+1)

empirical_means = sims_cumsum / n_arr


# ### (b)

# In[13]:


fair_die_prob = np.float64(1/6)

# Calculates expected value for 6 sided fair die
expected_value = np.sum(np.array(die_outcomes)*fair_die_prob)

abs_discrepency = np.abs(empirical_means - expected_value)


# ### (c)
# 
# 

# In[14]:


plt.figure(figsize=(15, 6))
plt.plot(n_arr, empirical_means, label="Empirical Average")
plt.axhline(y=expected_value, color='r', linestyle='--', label=f'Expected Value E[D]={expected_value}')
plt.xlabel("Sample Size n")
plt.ylabel("Empirical Average")
plt.title("Empirical Average vs Sample Size (n)")
plt.legend()
plt.grid(True)
plt.show()


# According to the plot, we can see that the sample average, converges to the true average or expected value of a fair dice roll as our sample size grows. As the **blue** line which is the sample average overlaps with the **red** line which is the true average for large sample sizes.

# ### (d)

# In[15]:


plt.figure(figsize=(15, 4))
plt.plot(n_arr, abs_discrepency, label="Absolute Discrepency")
plt.xlabel("Sample Size n")
plt.ylabel("Absolute Discrepency")
plt.title("Absolute Discrepency Between Empirical Average and E[D] vs Sample Size (n)")
plt.legend()
plt.grid(True)
plt.show()


# According to **Law of large numbers (LLN)**, states that the average of the results obtained from a large number of independent random samples converges to the true value, if it exists. Thus LLN guarantees that the empirical average of a random sample will converge to its expected value as the sample size grows. 
# 
# This phenomenon can be observed in our dice simulation, where the absolute difference between sample average and the true average or expected value of a fair dice roll, can be seen diminishing as our sample size grows. As the **blue** line which is the absolute difference converges towards zero for large sample sizes, and thus essentially, sample average and true average becoming equal.

# ## Problem 2.2

# ### (a)

# In[16]:


die_outcomes = [1,2,3,4,5,6]
die_probs = [0.1,0.1,0.1,0.1,0.1,0.5]
n_samples = 10000

sims = np.random.choice(die_outcomes,size=n_samples,p=die_probs)

# Calculates cumulative sum i.e calculates Di for all i in (1,n_samples)
sims_cumsum = np.cumsum(sims)

# Creates array sample size n for Di for all i in (1,n_samples)
n_arr = np.arange(1,n_samples+1)

empirical_means = sims_cumsum / n_arr


# ### (b)

# In[17]:


# Calculates expected value for unfair die with the given probabilities
expected_value = np.sum(np.array(die_outcomes)*np.array(die_probs))

abs_discrepency = np.abs(empirical_means - expected_value)


# ### (c)

# In[18]:


plt.figure(figsize=(15, 10))
plt.plot(n_arr, empirical_means, label="Empirical Average")
plt.axhline(y=expected_value, color='r', linestyle='--', label=f'Expected Value E[D]={expected_value}')
plt.xlabel("Sample Size n")
plt.ylabel("Empirical Average")
plt.title("Empirical Average vs Sample Size (n)")
plt.legend()
plt.grid(True)
plt.show()


# According to the plot, we can again see a similar trend as what we saw in case of fair dice simulation, even in case of unfair dice simulation we can see that the sample average, converges to the true average or expected value of a fair dice roll as our sample size grows. As the **blue** line which is the sample average overlaps with the **red** line which is the true average for large sample sizes. Thus this convergence happens irrespective of the probabilities of the outcomes in an experiment.

# ### (d)

# In[19]:


plt.figure(figsize=(15, 4))
plt.plot(n_arr, abs_discrepency, label="Absolute Discrepency")
plt.xlabel("Sample Size n")
plt.ylabel("Absolute Discrepency")
plt.title("Absolute Discrepency Between Empirical Average and E[D] vs Sample Size (n)")
plt.legend()
plt.grid(True)
plt.show()


# Here in this plot, we can again see a similar trend as what we saw in case of fair dice simulation, even in case of unfair dice simulation, we can see that the absolute difference between sample average and the true average or expected value of a fair dice roll, can be seen diminishing as our sample size grows. As the blue line which is the absolute difference converges towards zero for large sample sizes, and thus essentially, sample average and true average becoming equal. Therefore, we can conclude that the **LLN** holds and is true, irrespective of the probabilities of outcomes in an experiment, and for any experiment, given large sample size LLN can be observed.
