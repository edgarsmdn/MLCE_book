#!/usr/bin/env python
# coding: utf-8

# # 2. kNN for (Q)SPR modeling ⚛️
# 
# <a href="https://githubtocolab.com/edgarsmdn/MLCE_book/blob/main/02_kNN_QSPR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# ## Goals of this exercise 🌟

# *   We will learn how to construct a simple SPR model using kNN
# *   We will learn the importance of data normalization (scaling)
# *   We will review the concepts of training and test split and cross-validation
# *   We will review some of the performance metrics for assesing classification models

# ## A quick reminder ✅

# Probably the simplest data-driven model that you can think of is K-nearest neighbours (kNN). It simply predicts future data as the average (or mode) of the "k" nearest neighbours of the queried point.
# 
# As simple as this idea might be, it works relatively good in various applications. One of them is the generation of (quantitative) structure-property relationships ((Q)SPR) models {cite}`yuan2019developing, shen2003development`. Whether the word "Quantitative" is sometimes included or not depends on whether the model in question is a regression model or a classification model. Do you remember the difference?
# 
# The key question in kNN is what do we consider a neighbour and what not? This indicates us that we need to define a sort of similarity or distance metric that allows us to distinguish neighbouring points from points that are far away. 
# 
# Common distance metrics use the different [mathematical norms](https://en.wikipedia.org/wiki/Norm_(mathematics)). For example, the Euclidean distance:
# 
# $$
#  d(\textbf{x}, \textbf{x'}) = \sqrt{\sum_i^D (x_i - x'_i)^2}
# $$
# 
# ```{figure} media/02_kNN/kNN.png
# :alt: kNN
# :width: 60%
# :align: center
# 
# Among the k-nearest neighbours (k=5), the majority of points are red 1s. Therefore, the queried point (green x) will be labeled as "red 1". Image taken from {cite}`murphy2022probabilistic`. 
# ```
# 
# 
# 

# Let's exemplify the use of kNN by constructing a SPR model that predicts the whether a molecule is mutagenic or not. 
# 
# Mutagenicity is the property of substances to induce genetic mutation. It is one of the most important environmental, health and safety (EHS) properties to check when dealing with novel chemicals (e.g., drugs or solvents). In this case, we are going to use the data of mutagenicity on Salmonella typhimurium (Ames test). This dataset was collected by the [Istituto di Ricerche Farmacologiche Mario Negri](https://www.marionegri.it/), merging experimental data from a benchmark dataset
# compiled by {cite}`hansen2009benchmark` from a collection of data made available
# by the [Japan Health Ministry](https://www.nihs.go.jp/dgm/amesqsar.html) within their Ames (Q)SAR project.

# Let's fist import some libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ## Get data 📚
# 
# We have previously computed some molecular descriptors that will serve as input to our model. However, this is also an important step to consider when facing a problem like this: what are the important inputs to model mutagenicity? how do we know if these pre-computed features are enough for modeling mutagenicity? can we generate relevant molecular features automatically? 🤔
# 
# Ok, let's use [pandas](https://pandas.pydata.org/) to import the data as a DataFrame...

# In[2]:


if 'google.colab' in str(get_ipython()):
  df = pd.read_csv("https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/mutagenicity_kNN.csv")
else:
  df = pd.read_csv("references/mutagenicity_kNN.csv")

df


# The library pandas has many useful functions for data analytics. For example, we can print the type of the data we have...

# In[3]:


df.dtypes    


# And have a look at the first rows of our data to see how it looks like

# In[4]:


df.head()


# We can access columns in the DataFrame by the column's name

# In[5]:


y_experimental = df['Experimental value']
y_experimental


# and access rows by index

# In[6]:


first_rows = df.iloc[:4]
first_rows


# If we would like to get the subset of data that is labeled as mutagenic (i.e., 'Experimenal value' equal to 1), we could do it like this

# In[7]:


mutagenic_data = df[df['Experimental value']==1]
mutagenic_data


# Let's now collect all the input features of our dataset

# In[8]:


X = df.drop(['Unnamed: 0', 'Id','CAS','SMILES','Status','Experimental value','Predicted value'],axis=1)
X


# ### Exercise - manipulate a DataFrame ❗❗
# 
# * How many molecules in our dataset have a `qed` less than 0.5?
# * What is the molecule with the largest molecular weight `MolWt`?
# * What is the average number of valance electrons `NumValenceElectrons` of the molecules in our dataset? 

# In[9]:


# Your code here


# ## Feature scaling 📏

# It is always a good practice to scale your data before starting modeling. This helps the training process of many machine learning models. This is specially true for kNN which works on distances! The distance between two points is naturally afected by the dimensions of the input space. Look for example at the Euclidean distance, if one dimension ranges from 0 to 10,000 and another ranges from 0 to 1, the former one will potentially impact the distance value much more!
# 
# We do not want this to happen. Therefore, we need to scale all input features in order to give the same importance to all dimensions regardless of their original scale.
# 
# Here, we will use the method known as [standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler). Here, we move the distribution of the data to have unit variance and a mean equal to zero.
# 
# $$
# \hat{\textbf{x}} = \frac{\textbf{x}-\mu_x}{\sigma_x}
# $$
# 
# Of course there are other [scaling methods](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e) are available and you might want to review them and check which one is better than the other in which conditions.

# In[10]:


from sklearn.preprocessing import StandardScaler


# We initialize our scaler function

# In[11]:


scaler = StandardScaler()


# and fit it to our data (i.e., get the mean vector $\mu_x$ and the standard deviation vector $\sigma_x$.

# In[12]:


scaler.fit(X)


# Now, let's scale our data

# In[13]:


X_hat = scaler.transform(X)
X_hat


# ### Exercise - read documentation ❗❗
# 
# * What are exactly the mean an standard deviation vectors that we used to scaled the data? Go to the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and learn how you can access these. Reading the documentation of a library is a super important skill to learn! 

# In[14]:


# Your code here


# ## Training and test split ✂️
# 
# Now that we have scaled the data, we have to make sure we split it. Remember, our goal is to create a model that **generalize well** to unseen data and not simply fit some seen data perfectly. To achieve this, our goal while splitting the data should be to ensure that the distribution of the test set is as close as possible to the expected distribution of the future data. Hence, often what we want is to make the training and test datasets to have a similar distribution.
# 
# For example, let's see if how many molecules in our dataset are mutagenic vs. non-mutagenic.

# In[15]:


y = df['Experimental value'].to_numpy()
perc_mutagenic = y.sum()/len(y)*100
print('Percentage of mutagenic molecules    :', perc_mutagenic)
print('Percentage of non-mutagenic molecules:', 100-perc_mutagenic)


# In this case, the proportion of mutagenic and non-mutagenic data is very similar. Then, we will go ahead and split the data randomly. However, when you have a much more imbalanced dataset, how would you split the data better? You can look for instance at what [stratified splitting](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) is and why is it important.
# 
# In summary, when splitting your data, always think about the distribution of your splits!

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_hat, y, test_size=0.2, random_state=0)


# In[18]:


print('Training points: ', len(y_train))
print('Training points: ', len(y_test))


# ### Exercise - splits distribution ❗❗
# 
# * Check what is the proportion of mutagenic molecules in your train and test set? Was the random splitting good?

# In[19]:


# Your code here


# ## kNN model 🏘️

# In[20]:


from sklearn.neighbors import KNeighborsClassifier


# We initialize the kNN model by specifying the parameter "k". Later, we will review some ways that help us in determining this parameter better. For now, let's set it to 3.

# In[21]:


knn = KNeighborsClassifier(n_neighbors=3)


# we train it

# In[22]:


knn.fit(X_train, y_train)


# we predict the test set

# In[23]:


y_pred = knn.predict(X_test)


# Let's now evaluate our kNN model!
# 
# We will use several metrics for classification, for a quick reminder on them check the [documentation](https://scikit-learn.org/stable/modules/classes.html#classification-metrics).

# In[24]:


from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay) 


# Let first look at the confusion matrix

# In[25]:


cm = confusion_matrix(y_test, y_pred)
cm


# to see a prettier confusion matrix

# In[26]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# Now, let's look at the other metrics

# In[27]:


print('{:<10}  {:<15}'.format('Accuracy:', accuracy_score(y_test, y_pred)))
print('{:<10}  {:<15}'.format('Precision:', precision_score(y_test, y_pred)))
print('{:<10}  {:<15}'.format('Recall:', recall_score(y_test, y_pred)))
print('{:<10}  {:<15}'.format('F1:', f1_score(y_test, y_pred)))


# ### Comparison to VEGA model

# What do you think about the metrics? Is the kNN model performing good or bad?
# Let's compare it to the predictions of a kNN model trained on this same dataset and published as part of the [VEGA platform](https://www.vegahub.eu/portfolio-item/vega-qsar/). In the VEGA model, they have choosen k=4 with a similarity threshold of 0.7 (according to an internal similarity metric) Why is such threshold important?
# 
# Also, the developers of the VEGA kNN model used the [leave-one-out](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation) approach to assess the performace of their model. Why is this approach specially suitable when using the kNN algorithm? 

# In[28]:


y_vega = df['Predicted value']
y_vega


# Notice that the type of `y_vega` is `object`. This means that potentially its elements have different data types! Let's use a for-loop to see if we can discover which elements are causing the trouble.

# In[29]:


weird_predictions_idxs = []
for i, label in enumerate(y_vega):
  try:
    int(label)
  except:
    print('{:<4}, {:<10}'.format(i, label))
    weird_predictions_idxs.append(i)


# Indeed!! 6 elements of the series are not integers, but rather strings! This is consistent with the reports of VEGA that say that 6 compounds in this database were not able to be predicted due to not complying with the minimum similarity threshold of 0.7.
# 
# This illustrates clearly the concept of defining an [applicability domain](https://en.wikipedia.org/wiki/Applicability_domain) of the models we develop. Specially for chemical engineering application! 
# 
# Let's now remove these molecules from the comparison...

# In[30]:


df_clean = df.drop(index=weird_predictions_idxs)


# now, let's get the cleaned values 😀

# In[31]:


y_vega_clean = df_clean['Predicted value'].astype(int)
y_vega_clean


# #### VEGA model

# In[32]:


y_clean = df_clean['Experimental value'].astype(int)

cm = confusion_matrix(y_clean, y_vega_clean)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print('{:<10}  {:<15}'.format('Accuracy:', accuracy_score(y_clean, y_vega_clean)))
print('{:<10}  {:<15}'.format('Precision:', precision_score(y_clean, y_vega_clean)))
print('{:<10}  {:<15}'.format('Recall:', recall_score(y_clean, y_vega_clean)))
print('{:<10}  {:<15}'.format('F1:', f1_score(y_clean, y_vega_clean)))


# #### Exercise - leave-one-out cross-validation ❗❗
# 
# * Can you now assess our previous kNN model configuration using the leave-one-out approach on the cleaned dataset?

# In[33]:


from sklearn.model_selection import LeaveOneOut

X_clean = df_clean.drop(['Unnamed: 0', 'Id','CAS','SMILES','Status','Experimental value','Predicted value'],axis=1)

loo = LeaveOneOut()
print('Number of folds: ', loo.get_n_splits(X_clean))


# In[34]:


y_pred_loo_our_model = np.zeros(y_clean.shape[0])
for i, (train_index, test_index) in enumerate(loo.split(X_clean)):
  # Get training data for the current fold
  X_train_loo = X_clean.iloc[train_index]
  y_train_loo = y_clean.iloc[train_index]

  # Get test data for the current fold
  X_test_loo = X_clean.iloc[test_index]

  # Train kNN
  # Your code here

  # Get prediction on the test molecule
  # Your code here

  # Store the prediction in `y_pred_loo_our_model`
  # Your code here


# In[35]:


cm = confusion_matrix(y_clean, y_pred_loo_our_model)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print('{:<10}  {:<15}'.format('Accuracy:', accuracy_score(y_clean, y_pred_loo_our_model)))
print('{:<10}  {:<15}'.format('Precision:', precision_score(y_clean, y_pred_loo_our_model)))
print('{:<10}  {:<15}'.format('Recall:', recall_score(y_clean, y_pred_loo_our_model)))
print('{:<10}  {:<15}'.format('F1:', f1_score(y_clean, y_pred_loo_our_model)))


# 
# ## Finding the best k in kNN 🔎
# 
# What about the parameter k? How do we find the best k for our model? Why the VEGA model developers used k=4? Let's try to answer this...
# 
# k is a hyperparameter of the model and should be chosen using a validation set.
# 
# ```{important}
# Remember to reserve your test set exclusively for assesing your model! Never use it for training or hyperparameter tuning!  
# ```

# In[36]:


import numpy as np
from tqdm.notebook import tqdm

num_ks = np.arange(1, 100, 2).astype(int)

X_train_hyp, X_valid, y_train_hyp, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

train_accuracy = []
valid_accuracy = []

for i in tqdm(range(len(num_ks))):
  knn = KNeighborsClassifier(n_neighbors=num_ks[i])
  knn.fit(X_train_hyp, y_train_hyp)

  pred_train = knn.predict(X_train_hyp)
  pred_valid  = knn.predict(X_valid)

  train_accuracy.append(1-accuracy_score(y_train_hyp, pred_train))
  valid_accuracy.append(1-accuracy_score(y_valid, pred_valid))


# In[37]:


plt.figure(figsize=(10,6))
plt.plot(num_ks, train_accuracy, 'bs--', label='Train')
plt.plot(num_ks, valid_accuracy, 'rx--', label='Validation')
plt.xlabel('k')
plt.ylabel('Misclasification rate')
plt.legend()
plt.show()


# This graph is pretty similar to the one that we saw on slide 9 of Lecture 2. Here, we can see the expected general trend of the performance curves. 
# 
# Which k do you think is the best?

# ### Cross-validation
# 
# Problem: if validation set is small and noisy, it might be misleading
# Idea: Increase the size of the validation set
# Problem: This would reduce the size of the training set
# 
# Then, let's use all data for training and validation using k-fold cross-validation!

# In[38]:


from sklearn.model_selection import cross_validate


# In[39]:


num_ks = np.arange(1, 50, 1).astype(int)

train_misclassification = []
valid_misclassification = []
for i in tqdm(range(len(num_ks))):
  knn = KNeighborsClassifier(n_neighbors=num_ks[i])
  cv_dict = cross_validate(knn, X_train, y_train, cv=10, 
                                 scoring='accuracy', return_train_score=True)
  
  k_fold_train_scored = cv_dict['train_score']
  k_fold_valid_scored = cv_dict['test_score']
  
  train_misclassification.append(1-k_fold_train_scored.mean())
  valid_misclassification.append(1-k_fold_valid_scored.mean())
  


# In[40]:


plt.figure(figsize=(10,6))
plt.plot(num_ks, train_misclassification, 'bs--', label='Train')
plt.plot(num_ks, valid_misclassification, 'rx--', label='Validation')
plt.xlabel('k')
plt.ylabel('Misclasification rate')
plt.legend()
plt.show()


# In[41]:


print('k with minimum validation misclassification: ', num_ks[np.argmin(valid_misclassification)])


# In[42]:


plt.figure(figsize=(10,6))
plt.plot(num_ks, valid_misclassification, 'rx--', label='Validation')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('Misclasification rate')
plt.legend()
plt.show()


# You can now go back to your implementation of leave-one-out and train your kNN model using the best k that you found. How does it perform compared to the VEGA kNN? Do you got a better model? Why?

# ## Challenge - kNN QSPR for predicting BCF 🥇
# 
# 
# Develop a kNN model for regression to predict the bioconcentration factor (BCF).
# 
# 

# In[43]:


if 'google.colab' in str(get_ipython()):
  df_bcf = pd.read_csv("https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/BCF_training.csv")
else:
  df_bcf = pd.read_csv("references/BCF_training.csv")


# In[44]:


# Your code here


# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
