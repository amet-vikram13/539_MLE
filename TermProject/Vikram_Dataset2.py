#!/usr/bin/env python
# coding: utf-8

# # # [Amet Vikram]
# # # ECE 539 : Term Project
# # # Dataset 2 : Air Quality

# In[124]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("AirQualityUCI.csv",sep=";", decimal=",",header=0)


# In[3]:


display(df.head())


# ## Data Analysis and Exploration

# <h4>Premilinary Analysis</h4>

# In[4]:


print(df.info())


# <p>Now from above analysis, we can observe that there are two columns that are <strong>completely empty</strong> and so we will drop these columns.</p>
# <p>There are <strong>two temporal features</strong> present in the dataset as <strong>Date</strong> and <strong>Time</strong>, we will convert these features into one column as <strong>DateTime</strong>, to carry out <strong>time series analysis.</strong></p>
# <p>There are <strong>no categorical columns</strong> present in the dataset.</p>
# <p>Additonally, there is no target feature in the dataset and so we will use the potential pollutant feature columns,&nbsp;<strong>CO(GT)&nbsp;</strong>and&nbsp;<strong>PT08.S1(CO)</strong>, as <strong>two separate target features.</strong></p>

# In[5]:


target_class = pd.Index(["CO(GT)", "PT08.S1(CO)"])

features_drop = pd.Index(["Unnamed: 15","Unnamed: 16"])

num_cols = df.columns[2:15]

df = df.drop(features_drop,axis=1)

print(num_cols)
print("\n")

print(target_class)
print("\n")


# In[6]:


display(df.head())


# <h4>Null value Analysis</h4>
# 
# <p>Now based on the dataset description, it can be noted that it is mentioned that <strong>"Missing values are tagged with -200 value"</strong>, and so we will have to first replace the missing values and then proceed with the analysis.</p>
# <p><strong>Bibliography:</strong> <a href="https://archive.ics.uci.edu/dataset/360/air+quality">Dataset description</a></p>

# In[7]:


df = df.replace(to_replace = -200, value = np.nan)


# In[8]:


display(df.info())


# In[9]:


plt.figure(figsize=(10,10))
sns.heatmap(df.isna(),cbar=False)
plt.show()


# In[10]:


samples_large_omissions = []
for i in range(len(df)):
    val = df.loc[i,:].isna().sum()
    if val > 0.3*len(df.columns) :
        samples_large_omissions.append(i)
print("Data Samples with more than 30% missing values (around 4 columns):",len(samples_large_omissions)," ",len(samples_large_omissions)/len(df))
print("\n")

samples_large_omissions = []
for i in range(len(df)):
    val = df.loc[i,:].isna().sum()
    if val > 0.9*len(df.columns) :
        samples_large_omissions.append(i)
print("Data Samples with more than 90% missing values (all columns):",samples_large_omissions)


# In[11]:


features_large_omissions = []
for i,col in enumerate(df.columns):
    val = df.loc[:,col].isna().sum()
    if val > 0.1*len(df) :
        features_large_omissions.append(col)
        print("{}) {} : {} {}".format(i,col,val,val/len(df)))
print("Features with more than 10% missing values:",features_large_omissions)


# <p>Now based on the above analysis we can make following points :-</p>
# <ul>
# <li>Roughly around 5% of data samples have atleast 4 columns missing.&nbsp;</li>
# <li>There are 4 feature columns for which roughly more 20% data samples are missing. We can use appropriate imputer for these feature columns.</li>
# <li>There is one feature column <strong>"NMHC(GT)"&nbsp;</strong>for which almost all feature data is missing and so we will drop this column.</li>
# <li>There are some data samples which have all feature columns missing that is the whole row is just empty and so we will drop these data samples.</li>
# </ul>

# In[12]:


df = df.drop(samples_large_omissions,axis=0)

df = df.drop(["NMHC(GT)"],axis=1)
num_cols = num_cols.drop("NMHC(GT)")

display(df.info())
display(df.head())


# <h4>Numerical Analysis</h4>

# In[145]:


display(df[num_cols].describe())


# In[147]:


fig, axes = plt.subplots(6, 2, figsize=(2 * 5, 6 * 5))
axes = axes.ravel()
for i,col in enumerate(num_cols):
    axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[13]:


display(df[num_cols].skew())


# In[167]:


fig, axes = plt.subplots(6, 2, figsize=(2 * 5, 6 * 5))
axes = axes.ravel()
for i,col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=20, color='green', alpha=0.7, edgecolor='black', ax=axes[i])
    axes[i].set_title("{} -- Skewnes = {}".format(col,round(df[col].skew(),4)))

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[152]:


arr = []
for col in num_cols:
    arr += df[col].dropna().to_list()

print(pd.Series(arr).describe())

fig, axes = plt.subplots(2,1,figsize=(1*8, 2*8))

axes[0].hist(arr, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title("All Data")

axes[1].boxplot(arr, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[1].set_title("All Data")

plt.tight_layout()
plt.show()


# <p>Now based on the above analysis we can make following points :-</p>
# <ul>
# <li>From the <strong>"box-plots"</strong>, we can observe that, some of the feature columns have large number of outliers.</li>
# <li>Upon examining the <strong>skewness, kde plots and histogram,</strong> we can conclude that the feature columns <strong>CO(GT), PT08.S1(CO) , C6H6(GT), NOx(GT), </strong>and<strong> PT08.S3(NOx), </strong>are highly positively skewed. We will use appropriate transformation to remove the skewness from these feature columns. <strong>[Refer bibliography]</strong></li>
# <li>Overall, the whole dataset does not exhibit any particular kind of trend.</li>
# </ul>
# <p><strong>Bibliography : </strong><a href="https://machinelearningmastery.com/skewness-be-gone-transformative-tricks-for-data-scientists/">Removing skewness</a></p>

# <h4>Correlation Analysis</h4>

# In[170]:


def get_top_abs_correlations(data=df,n=5):
    corr = df[num_cols].corr().abs().unstack()
    pairs_to_drop = set()
    for i in range(len(num_cols)):
        for j in range(0,i+1):
            pairs_to_drop.add((num_cols[i],num_cols[j]))
    corr = corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return corr[:n]

print(get_top_abs_correlations(df,np.square(len(num_cols))))


# In[185]:


all_corr_count = np.square(len(num_cols))
corr = get_top_abs_correlations(df,all_corr_count)

corr_idx = []
for i,val in enumerate(corr.index):
    x = val[0]
    y = val[1]
    if x in target_class or y in target_class:
        corr_idx.append(i)

corr = corr[corr.index[corr_idx]]
n = len(corr)

fig, axes = plt.subplots(n, 2, figsize=(2 * 7, n * 7))

axes = axes.ravel()

for i,val in enumerate(corr.index):
    x = val[0]
    y = val[1]
    sns.scatterplot(x=df[x],y=df[y],ax=axes[i]);
    axes[i].set_title(y+" -- "+x)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[177]:


all_corr_count = np.square(len(num_cols))
corr = get_top_abs_correlations(df,all_corr_count)

print(corr.describe())

fig, axes = plt.subplots(figsize=(8, 8))

axes.boxplot(corr, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes.set_title("Correlation Data")

plt.tight_layout()
plt.show()

print("No of pair of features with correlation > 0.8 are: {}".format(sum(corr>0.8)))
print("No of pair of features with correlation > 0.5 are: {}".format(sum(corr>0.5)))


# In[179]:


plt.figure(figsize=(10,5))
sns.heatmap(df[num_cols].corr(),cmap='YlGnBu',annot=True)
plt.show()


# In[215]:


X = df[num_cols].copy()

X = X.dropna()

X = PowerTransformer(method="yeo-johnson").fit_transform(X)

X = pd.DataFrame(X,columns=num_cols)

X = X.drop(target_class[0],axis=1)

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)
print("\n")

X = df[num_cols].copy()

X = X.dropna()

X = PowerTransformer(method="yeo-johnson").fit_transform(X)

X = pd.DataFrame(X,columns=num_cols)

X = X.drop(target_class[1],axis=1)

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)


# <p>Now based on the above analysis we can make following points :-</p>
# <ul>
# <li>Based on the scatter plots, we can conclude that there is not enough evidence that our target feature variables, "<strong>CO(GT)</strong>" and "<strong>PT08.S1(CO)</strong>", are strongly correlated to most of the other feature variables.</li>
# <li>Upong conducting the <strong>"Variation Inflation Factor"</strong> test, we can observe that the feature columns, <strong>"C6H6(GT)"</strong> and <strong>"PT08.S2(NMHC)",</strong> have very high VIF values indicating<strong> high multicollinearity</strong> in the data.</li>
# <li>Techniques like <strong>"lasso"</strong> and <strong>"ridge"</strong> regression can be used to address this multicollinearity and penalize the regressors which causes the same.<strong>[Refer Bibliography]</strong></li>
# <li>For overall data, examining the <strong>"box-plots"</strong> and <strong>"heatmap"</strong>, we can conclude that there is evidence of strong correlation between the <strong>"pollutant feature columns".</strong></li>
# </ul>
# <p><strong>Bibliography: <a href="https://medium.com/@satyarepala/tackling-multicollinearity-understanding-variance-inflation-factor-vif-and-mitigation-techniques-2521ebf024b6">Addressing multicollinearity</a></strong></p>

# <h4>Temporal Analysis</h4>

# In[14]:


timestamp = (df["Date"]) + " " + (df["Time"])
timestamp = timestamp.apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H.%M.%S"))

month = timestamp.dt.month_name()
day = timestamp.dt.day_name()
hour = timestamp.dt.hour

df_temp = pd.DataFrame({
    'DateTime': timestamp,
    'Month': month,
    'Day': day,
    'Hour': hour
})
df = pd.concat([df_temp, df], axis=1)

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

df = df.drop(["Time"],axis=1)

display(df.info())
print("\n")
display(df.head())


# In[256]:


month_df_list = []
day_df_list   = []
hour_df_list  = []

months = ['January','February','March', 'April', 'May','June', 
          'July', 'August', 'September', 'October', 'November', 'December']

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for month in months:
    temp_df = df.loc[(df["Month"] == month)]
    month_df_list.append(temp_df)

for day in days:
    temp_df = df.loc[(df["Day"] == day)]
    day_df_list.append(temp_df)

for hour in range(24):
    temp_df = df.loc[(df["Hour"] == hour)]
    hour_df_list.append(temp_df)

def df_time_plotter(df_list, time_unit, y_col):
    if time_unit == 'M':
        nRows = 3
        nCols = 4
        n_iter = len(months)
    elif time_unit == 'D':
        nRows = 2
        nCols = 4
        n_iter = len(days)
    elif time_unit == 'H':
        nRows = 4
        nCols = 6
        n_iter = 24
        
    fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize = (40,30))
    axs = axs.ravel()
    for i in range(n_iter):
        data = df_list[i]
        ax = axs[i]
        data.plot(kind ='scatter', x = 'DateTime', y= y_col , ax = ax, fontsize = 24)
        ax.set_ylabel('Pollutant Concentration',fontsize=30)
        ax.set_xlabel('')
        if time_unit == 'M':
            ax.set_title(y_col + ' ' + months[i],  size=40) # Title
        elif time_unit == 'D':
            ax.set_title(y_col + ' ' + days[i],  size=40) # Title
        else:
             ax.set_title(y_col + ' ' + str(i),  size=40) # Title
        ax.tick_params(labelrotation=60)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.5)
    plt.show()


# In[258]:


df_time_plotter(month_df_list,"M","PT08.S1(CO)")


# In[259]:


df_time_plotter(day_df_list,"D","PT08.S1(CO)")


# In[260]:


df_time_plotter(hour_df_list,"H","PT08.S1(CO)")


# In[262]:


sns.barplot(x = 'Month', y = 'PT08.S1(CO)', data = df)
plt.title('CO Values Per Month')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x = 'Day', y = 'PT08.S1(CO)', data = df)
plt.title('CO Values Per Day of the Week')
plt.xticks(rotation=90)
plt.show()

sns.barplot(x = 'Hour', y = 'PT08.S1(CO)', data = df)
plt.title('CO Values Per Hour')
plt.xticks(rotation=90)
plt.show()


# 

# # Data Preprocessing

# In[18]:


features_drop = df.columns.drop(num_cols)

print("Taget feature is : {}".format(target_class))
print("\n")

print("Features to drop are:-")
print(features_drop)
print("\n")

print("Numerical Columns are:-")
print(num_cols)


# In[70]:


class FeatureDrop(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.drop(self.features,axis=1)
        return X

drop_feat = FeatureDrop(features_drop)
mean_impute = SimpleImputer(strategy="mean")
std_scaler = StandardScaler()
skew_scale = PowerTransformer(method="yeo-johnson")


# <h4> Train Test Split </h4>

# In[23]:


df_train, df_test = train_test_split(df,stratify=df["Day"],test_size=0.2, random_state=42)

print("Length of df_train is : {}".format(len(df_train)))
print(df_train["Day"].value_counts())
print("\n")

print("Length of df_test is : {}".format(len(df_test)))
print(df_test["Day"].value_counts())
print("\n")


# # Model Training and Evaluation

# <p>Based on above all the analysis, we will use 3 kind of models to train and evaluate our data. These models are :-</p>
# <ol>
# <li><strong>Ridge regression&nbsp;</strong></li>
# <li><strong>Lasso&nbsp;regression</strong></li>
# <li><strong>ElasticNet regression</strong></li>
# </ol>
# <p>Now the model training plan would be similar for all the three models and is outlined below :-</p>
# <ul>
# <li><strong>model&nbsp;= {"Feature Drop", "Data Imputation", "Skew Balance", "Standard Normalization", "Parameter Tuning", "Training", "Evaluation" }</strong></li>
# </ul>
# <p>We will have <strong>two regression models</strong> for each for one of the target class feature columns.</p>

# In[133]:


def evaluation(y_pred,y_test):
    r2 = r2_score(y_test, y_pred)
    mas = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    
    print("R2 score = ", r2_score(y_test, y_pred))
    print("\n")

    print("MAS = ", mean_absolute_error(y_test,y_pred))
    print("\n")

    print("MSE = ", mean_squared_error(y_test,y_pred))
    print("\n")

    return [r2, mas, mse]

all_scores = []


# <h4> Ridge Regression </h4>

# In[134]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

ridge_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(ridge_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(ridge_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha":np.linspace(0.01,0.1,20)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = Ridge(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[135]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

ridge = Ridge(alpha=0.1)

ridge_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(ridge_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(ridge_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(ridge, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

ridge.fit(X_train,y_train)


# In[136]:


y_pred = ridge.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# In[137]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

ridge_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(ridge_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(ridge_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha":np.linspace(0.01,0.1,20)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = Ridge(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[138]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

ridge = Ridge(alpha=0.04)

ridge_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(ridge_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(ridge_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(ridge, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

ridge.fit(X_train,y_train)


# In[139]:


y_pred = ridge.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# <h4>Lasso Regression</h4>

# In[140]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

lasso_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(lasso_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(lasso_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha":np.linspace(0.01,0.1,20)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = Lasso(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[141]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

lasso = Lasso(alpha=0.01)

lasso_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(lasso_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(lasso_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(lasso, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

lasso.fit(X_train,y_train)


# In[142]:


y_pred = lasso.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# In[143]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

lasso_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(lasso_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(lasso_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha":np.linspace(0.01,0.1,20)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = Lasso(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[144]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

lasso = Lasso(alpha=0.01)

lasso_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(lasso_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(lasso_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(lasso, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

lasso.fit(X_train,y_train)


# In[145]:


y_pred = lasso.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# <h4>ElasticNet Regression</h4>

# In[146]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

elasticnet_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(elasticnet_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(elasticnet_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],"l1_ratio": np.arange(0.0, 1.0, 0.1)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = ElasticNet(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[147]:


response = target_class[0]

X_train = df_train.copy()
X_test = df_test.copy()

elastic_net = ElasticNet(alpha=0.0001,l1_ratio=0.0)

elasticnet_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(elasticnet_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(elasticnet_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(elastic_net, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

elastic_net.fit(X_train,y_train)


# In[148]:


y_pred = elastic_net.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# In[149]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

elasticnet_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(elasticnet_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(elasticnet_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

hyper_parameters = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],"l1_ratio": np.arange(0.0, 1.0, 0.1)}
scoring = ["r2","neg_root_mean_squared_error"]
for metric in scoring:
  clf = GridSearchCV(estimator = ElasticNet(),param_grid=hyper_parameters,scoring = metric)

  clf.fit(X_train,y_train)
  
  print("Best param for ridge are:-")
  print(clf.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf.best_score_))
  print("\n")


# In[150]:


response = target_class[1]

X_train = df_train.copy()
X_test = df_test.copy()

elastic_net = ElasticNet(alpha=0.0001,l1_ratio=0.0)

elasticnet_pipline = make_pipeline(drop_feat,mean_impute,skew_scale,std_scaler)

X_train = pd.DataFrame(elasticnet_pipline.fit_transform(X_train),columns=num_cols)
X_test = pd.DataFrame(elasticnet_pipline.fit_transform(X_test),columns=num_cols)

y_train = X_train[response].copy()
X_train = X_train.drop(response,axis=1)

y_test = X_test[response].copy()
X_test = X_test.drop(response,axis=1)

N, train_score, val_score = learning_curve(elastic_net, X_train, y_train, cv=4, scoring="r2", train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

elastic_net.fit(X_train,y_train)


# In[151]:


y_pred = elastic_net.predict(X_test)

all_scores.append(evaluation(y_pred,y_test))
print("\n")


# <h4> Conclusion </h4>

# In[161]:


all_scores_pollutant_1 = all_scores[0::2]
all_scores_pollutant_2 = all_scores[1::2]

all_scores_pollutant_1 = pd.DataFrame(np.array(all_scores_pollutant_1),columns=["R2","MAS","MSE"],index=["Ridge","Lasso","ElasticNet"])
all_scores_pollutant_2 = pd.DataFrame(np.array(all_scores_pollutant_2),columns=["R2","MAS","MSE"],index=["Ridge","Lasso","ElasticNet"])

print("Pollutant 1: {}".format(target_class[0]))
display(all_scores_pollutant_1)
print("\n")
print("Pollutant 2: {}".format(target_class[1]))
display(all_scores_pollutant_2)


# In[ ]:




