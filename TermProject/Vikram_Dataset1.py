#!/usr/bin/env python
# coding: utf-8

# # # [Amet Vikram]
# # # ECE 539 : Term Project
# # # Dataset 1 : Mice Protein Expression

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Data_Cortex_Nuclear.csv")


# In[3]:


display(df.head())


# ## Data Analysis and Exploration

# <h4>Premilinary Analysis</h4>

# In[4]:


print(df.info())


# <p>Now from the above analysis we can make the following points :-</p>
# <ul>
# <li>We can observe that columns from index <strong>1 to 77</strong> are all <strong>"float64"</strong> that is <strong>numerical type</strong> and columns from index <strong>78 to 81</strong> are all <strong>"object"</strong> that is <strong>non-numerical or categorical type.</strong></li>
# <li>We will seperate these columns based on the numerical and categorical nature into two seperate lists for further analysis and processing.</li>
# <li>We can also drop the first column <strong>(index 0) "MouseID"</strong> as it has no significance in our analysis and prediction.</li>
# </ul>

# In[4]:


target_class = "class"

features_drop = pd.Index(["MouseID"])

num_cols = df.columns[1:78]
cat_cols = df.columns[78:]

print(num_cols)
print("\n")

print(cat_cols)
print("\n")

print("Features to drop are:-")
print(features_drop)
print("\n")

print("Target Feature:",target_class)


# <h4>Null value Analysis</h4>

# In[6]:


plt.figure(figsize=(10,10))
sns.heatmap(df.isna(),cbar=False)
plt.show()


# In[5]:


samples_large_omissions = []
for i in range(len(df)):
    val = df.loc[i,:].isna().sum()
    if val > len(df.columns)/4 :
        samples_large_omissions.append(i)
        print(i,":",val)
print("Data Samples with more than 25% missing values:",samples_large_omissions)


# In[6]:


features_large_omissions = []
for i,col in enumerate(df.columns):
    val = df.loc[:,col].isna().sum()
    if val > 0.125*len(df) :
        features_large_omissions.append(col)
        print("{}) {} : {}".format(i,col,val))
print("Features with more than 12.5% missing values:",features_large_omissions)


# <p>Now from the above analysis and heatmap of the missing values we can make the following points :-</p>
# <ul>
# <li>Overall the dataset does not have too many missing values.</li>
# <li>For&nbsp;3 data samples, there is more than 25% of feature columns data missing, <strong>43 feature columns data out of 82 columns</strong> to be precise.</li>
# <li>For 5 features columns, specifically from index <strong>69 to 76</strong>, there are more than 12.5% data samples that does not have information have for these 5 feature columns.</li>
# </ul>
# <p>With the above points, we can conclude that <strong>Data Imputation</strong>&nbsp;is required and we can use appropriate imputation method to fill the gaps in the data samples mentioned in above points.</p>

# <h4>Categorical value Analysis</h4>

# In[9]:


for col in cat_cols:
    print(col,":",pd.unique(df[col]))


# In[10]:


fig, axes = plt.subplots(4, 1, figsize=(10, 4 * 7))

for i,col in enumerate(cat_cols):
    sns.countplot(x=col, data=df, ax=axes[i]);
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# <p>Now from the above analysis and bar plots of the categorical values we can make the following points :-</p>
# <ul>
# <li>Overall for all 3 categorical values -- <strong>Genotype, Treatment, and Behabiour</strong>, we have nearly equal number of data sample for each class</li>
# <li>For the <strong>"class"</strong> feature, we have a slight <strong>class imbalance</strong> with respect to <strong>"t-CS-s"</strong> class where we have relatively a smaller set of data samples as compared to other classes.</li>
# <li><strong>NOTE:</strong> Upon examining the categorical features, it is clear that the target feature is a combination of categorical feature in the format <strong>Genotype-Behabiour-Treatment</strong>, and so we will use these categorical features as part of our target columns to conduct any analysis and training of the model.</li>
# <li>All categorical features have <strong>no inherent order,</strong> and so a <strong>nominal encoding</strong> method can be employed to encode all the categorical variables.</li>
# <li>It should be noted while using nominal encoding methods such as <strong>"One hot encoding",&nbsp;</strong>we are essentially increasing the dimension of our data and since the dimension of data is already high enough, we might have to use some kind <strong>"Dimensional reduction method"</strong> to get the best results along with <strong>"One hot encoding". [Refer bibliography]</strong></li>
# </ul>
# <p>With the above points, we can conclude that there is <strong>no major class imbalance</strong> in the dataset for any categorical value and we have to use <strong>stratified sampling methods</strong> to separate our training and test data and then use appropraite training algorithms to train the data and <strong>transform the&nbsp;categorical&nbsp;variables using nominal encoding method.</strong></p>
# <p><strong>Bibliography: <a href="https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor/9447#9447">LabelEncoder vs OHE</a></strong></p>

# <h4>Numerical value Analysis</h4>

# In[11]:


display(df[num_cols].describe())


# In[49]:


fig, axes = plt.subplots(39, 2, figsize=(15, 15 * 10))
axes = axes.ravel()
for i,col in enumerate(num_cols):
    axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[48]:


fig, axes = plt.subplots(39, 2, figsize=(15, 15 * 10))
axes = axes.ravel()
for i,col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[87]:


arr = []
for col in num_cols:
    arr += df[col].dropna().to_list()

print(pd.Series(arr).describe())

fig, axes = plt.subplots(2,1,figsize=(1*8, 2*8))

axes[0].hist(arr, bins=30, density=True, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title("All Data")

axes[1].boxplot(arr, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[1].set_title("All Data")

plt.tight_layout()
plt.show()


# <p>Now based on the above analysis we can make following points :-</p>
# <ul>
# <li>From the <strong>"box-plots"</strong>, we can observe that, most of the feature columns have moderate amount of outliers and so wont impact have much impact on classification algorithms. The <strong>"Standard Normalization"</strong> can handle the outliers and will also aid in model training for some algorithms.</li>
# <li>Upon examining the histograms for each numerical feature, we can notice one very important observation that for maximum of the feature columns, the histogram exhibit somewhat of <strong>"Gaussian"</strong> nature and thus, <strong>"Gaussian Naive Bayes"</strong> can be employed for this dataset.</li>
# <li>Overall, the whole dataset does not exhibit any <strong>"Gaussian"</strong> nature, and so algorithms like <strong>"QDA"</strong> cannot be employed here. Althought the data is somewhat displaying a <strong>"Poisson Distribution"</strong> nature, but there are no classification algorithms that can leverage the poisson process.</li>
# </ul>

# <h4>Correlation Analysis</h4>

# In[76]:


def get_top_abs_correlations(data=df,n=5):
    corr = df[num_cols].corr().abs().unstack()
    pairs_to_drop = set()
    for i in range(len(num_cols)):
        for j in range(0,i+1):
            pairs_to_drop.add((num_cols[i],num_cols[j]))
    corr = corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return corr[:n]

print(get_top_abs_correlations(df,30))


# In[74]:


corr = get_top_abs_correlations(df,20)

fig, axes = plt.subplots(10, 2, figsize=(2 * 7, 10 * 7))

axes = axes.ravel()

for i,val in enumerate(corr.index):
    x = val[0]
    y = val[1]
    sns.scatterplot(x=df[x],y=df[y],hue=df[target_class],ax=axes[i]);
    axes[i].set_title(y+" -- "+x)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[97]:


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


# <p>Now based on the above analysis we can make following points :-</p>
# <ul>
# <li>Based on the<strong> top 30</strong> correlations data and <strong>scatter plots</strong>, we can conclude that there is strong evidence of high correlation among feature data.</li>
# <li>The <strong>box-plot</strong> suggests that for the <strong>overall correlation data</strong>, there is roughly around 60% pair of features with very <strong>high correlation</strong> and thus indicating <strong>high multicollinearity.</strong></li>
# <li>We will have to use <strong>PCA analysis</strong> to address this multicollinearity in data for efficient training of algorithms.</li>
# </ul>

# # Data Preprocessing

# In[7]:


training_features = num_cols

print("Taget feature is : {}".format(target_class))
print("\n")

print("Features to drop are:-")
print(features_drop)
print("\n")

print("Categorical Columns are:-")
print(cat_cols)
print("\n")

print("Numerical Columns are:-")
print(num_cols)


# In[8]:


class FeatureDrop(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.drop(self.features,axis=1)
        return X

drop_feat = FeatureDrop(features_drop.join(cat_cols,how="outer"))
mean_impute = SimpleImputer(strategy="mean")
std_scaler = StandardScaler()
onehot_enc = OneHotEncoder()
label_enc = LabelEncoder()


# 

# <h4> Train Test Split </h4>

# In[9]:


df_train, df_test = train_test_split(df,stratify=df[target_class],test_size=0.2, random_state=42)

print("Length of df_train is : {}".format(len(df_train)))
print(df_train[target_class].value_counts())
print("\n")

print("Length of df_test is : {}".format(len(df_test)))
print(df_test[target_class].value_counts())
print("\n")


# # Feature Transformation : PCA Analysis

# In[10]:


X_train = df_train.copy()

pca_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler)


# In[11]:


display(pca_pipeline)


# In[12]:


X_train = pd.DataFrame(pca_pipeline.fit_transform(X_train),columns=num_cols)


# In[13]:


display(X_train.head())


# <h4>Choosing optimal number of PCA components</h4>

# In[14]:


pcascree = PCA()

pcascree.fit(X_train)

# Get the explained variance ratio for each principal component
exp_var_ratio = pcascree.explained_variance_ratio_

# Create a scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(exp_var_ratio) + 1), exp_var_ratio, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()


# Like Scree plot suggests there must be at least 10 principal components, and so we can conclude that the maximum of the variance is being explained by the first 10 principal components.

# # Model Training and Evaluation

# <p>Based on above all the analysis, we will use 3 kind of models to train and evaluate our data. These models are :-</p>
# <ol>
# <li><strong>Gaussian Naive Bayes</strong></li>
# <li><strong>KNN classification</strong></li>
# <li><strong>SVM</strong></li>
# </ol>
# <p>Now the model training plan is outlined below :-</p>
# <ol>
# <li><strong>GNB = {"Feature Drop", "Data Imputation", "Training", "Evaluation" }</strong> -&gt; We will use only target_class feature column for this model.</li>
# <li><strong>KNN = {"Feature Drop", "Data Imputation", "Standard scaling", "PCA transformation", "Training", "Parameter Tuning", "Evaluation" }</strong> -&gt; We will use only target_class feature column for this model.</li>
# <li><strong>SVM ={"Feature Drop", "Data Imputation", "Standard scaling", "Training", "Parameter Tuning", "Evaluation" }</strong> -&gt; We will use 3 SVM models here for each one of target categorical features and compare the result with the final target feature.</li>
# </ol>

# In[29]:


def evaluation(y_pred,y_test):
    print("Accuracy = ", accuracy_score(y_test, y_pred))
    print("\n")

    print("The confusion matrix is as follows:-")
    print(confusion_matrix(y_test,y_pred))
    print("\n")

    print("The classification report is as follows:-")
    print(classification_report(y_test,y_pred))
    print("\n")


# <h4> Gaussian Naive Bayes </h4>

# In[16]:


X_train, y_train = df_train.copy(), df_train[target_class].copy()

X_test, y_test = df_test.copy(), df_test[target_class].copy()


gnb = GaussianNB()

gnb_pipeline = make_pipeline(drop_feat, mean_impute, gnb)

gnb_pipeline.fit(X_train,y_train)


# In[17]:


y_pred = gnb_pipeline.predict(X_test)

print(evaluation(y_pred,y_test))


# <h4> K-Nearest Neighbour</h4>

# In[18]:


knn_train, knn_val = train_test_split(df_train,test_size=0.2, random_state=42)

X_train, y_train = knn_train.copy(), knn_train[target_class].copy()

X_val, y_val = knn_val.copy(), knn_val[target_class].copy()

pca = PCA(n_components = 10)

knn_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler,pca)

X_train = knn_pipeline.fit_transform(X_train)
X_val = knn_pipeline.transform(X_val)

k_values = range(1, 10)
training_errors = []
testing_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_val = knn.predict(X_val)
    training_errors.append(1 - accuracy_score(y_train, y_pred_train))
    testing_errors.append(1 - accuracy_score(y_val, y_pred_val))

plt.figure(figsize=(10, 6))
plt.plot(k_values, training_errors, marker='o', linestyle='--', color='red', label='Training Error')
plt.plot(k_values, testing_errors, marker='o', linestyle='--', color='blue', label='Validation Error')
plt.xticks(k_values)
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate')
plt.legend()
plt.grid()
plt.show()


# https://stackoverflow.com/questions/68284264/does-the-pipeline-object-in-sklearn-transform-the-test-data-when-using-the-pred

# In[19]:


X_train, y_train = df_train.copy(), df_train[target_class].copy()

X_test, y_test = df_test.copy(), df_test[target_class].copy()

pca = PCA(n_components = 10)
knn = KNeighborsClassifier(n_neighbors=3)

knn_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler,pca,knn)

N, train_score, val_score = learning_curve(knn_pipeline, X_train, y_train, cv=4, scoring='accuracy', train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

knn_pipeline.fit(X_train, y_train)


# In[20]:


y_pred = knn_pipeline.predict(X_test)

print(evaluation(y_pred,y_test))


# <h4> SVM </h4>

# In[21]:


X_train, y_train = df_train.copy(), df_train[cat_cols[:-1]].copy()

y_train_0 = label_enc.fit_transform(y_train[cat_cols[0]])
y_train_1 = label_enc.fit_transform(y_train[cat_cols[1]])
y_train_2 = label_enc.fit_transform(y_train[cat_cols[2]])

hyper_parameters = {"kernel":("rbf","linear","poly"),"C":[.001,.01,1,10],"degree":[2,3,5,8]}

svm_0 = SVC(probability=True)
svm_1 = SVC(probability=True)
svm_2 = SVC(probability=True)

svm_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler)

X_train = svm_pipeline.fit_transform(X_train)

scoring = ['balanced_accuracy','f1','roc_auc']
for metric in scoring:
  clf_0 = GridSearchCV(estimator = svm_0,param_grid=hyper_parameters,scoring = metric)
  clf_1 = GridSearchCV(estimator = svm_1,param_grid=hyper_parameters,scoring = metric)
  clf_2 = GridSearchCV(estimator = svm_2,param_grid=hyper_parameters,scoring = metric)
  
  clf_0.fit(X_train,y_train_0)
  clf_1.fit(X_train,y_train_1)
  clf_2.fit(X_train,y_train_2)
  
  print("Best param for {} svm are:-".format(cat_cols[0]))
  print(clf_0.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf_0.best_score_))
  print("\n")
  print("Best param for {} svm are:-".format(cat_cols[1]))
  print(clf_1.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf_1.best_score_))
  print("\n")
  print("Best param for {} svm are:-".format(cat_cols[2]))
  print(clf_2.best_params_)
  print("\n")
  print("Best {} : {}".format(metric,clf_2.best_score_))
  print("\n")
  print("\n")


# In[26]:


X_train, y_train = df_train.copy(), df_train[cat_cols[:-1]].copy()

X_test, y_test = df_test.copy(), df_test[cat_cols[:-1]].copy()

y_train_0 = y_train[cat_cols[0]]
y_train_1 = y_train[cat_cols[1]]
y_train_2 = y_train[cat_cols[2]]

y_test_0 = y_test[cat_cols[0]]
y_test_1 = y_test[cat_cols[1]]
y_test_2 = y_test[cat_cols[2]]

svm_0 = SVC(C=10,degree=2,kernel="rbf",probability=True)
svm_1 = SVC(C=1,degree=2,kernel="rbf",probability=True)
svm_2 = SVC(C=0.01,degree=2,kernel="linear",probability=True)

svm0_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler,svm_0)
svm1_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler,svm_1)
svm2_pipeline = make_pipeline(drop_feat,mean_impute,std_scaler,svm_2)

N, train_score, val_score = learning_curve(svm0_pipeline, X_train, y_train_0, cv=4, scoring='accuracy', train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

N, train_score, val_score = learning_curve(svm1_pipeline, X_train, y_train_1, cv=4, scoring='accuracy', train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

N, train_score, val_score = learning_curve(svm2_pipeline, X_train, y_train_2, cv=4, scoring='accuracy', train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

svm0_pipeline.fit(X_train, y_train_0)
svm1_pipeline.fit(X_train, y_train_1)
svm2_pipeline.fit(X_train, y_train_2)


# In[27]:


y_pred_0 = svm0_pipeline.predict(X_test)
y_pred_1 = svm1_pipeline.predict(X_test)
y_pred_2 = svm2_pipeline.predict(X_test)

print(evaluation(y_pred_0,y_test_0))
print("\n")
print(evaluation(y_pred_1,y_test_1))
print("\n")
print(evaluation(y_pred_2,y_test_2))


# 
