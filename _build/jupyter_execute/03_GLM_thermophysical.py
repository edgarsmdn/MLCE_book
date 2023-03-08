#!/usr/bin/env python
# coding: utf-8

# # 3. GLM for thermophysical property prediction ⚗️
# 
# <a href="https://githubtocolab.com/edgarsmdn/MLCE_book/blob/main/03_GLM_thermophysical.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# ## Goals of this exercise 🌟

# * We will learn how to apply (generalized) linear regression
# * We will review some performance metrics for assesing regression models

# ## A quick reminder ✅

# The process of "learning" in the context of supervised **learning** can be seen as exploring a hypothesis space $\mathcal{H}$ looking for the most appropriate hypothesis function $h$. In the context of linear regression the hypothesis space is of course the space of linear functions.
# 
# Let's imagine our input space is two-dimensional, continuous and non-negative. This could be denoted mathematically as $\textbf{x} \in \mathbb{R}_+^2$. For example, for an ideal gas, its pressure is a function of the temperature and volume. In this case, our dataset will be a collection of $N$ points with temperature and volume values as inputs and pressure as output
# 
# $$
# \{(\textbf{x}^{(1)}, y^{(1)}), (\textbf{x}^{(2)}, y^{(2)}), ..., (\textbf{x}^{(N)}, y^{(N)}) \}
# $$
# 
# where for each point $\textbf{x} = [x_1, x_2]^T$. Our hypothesis function would be
# 
# $$
# h(\textbf{x}, \textbf{w}) = w_0 + w_1x_1 + w_2+x_2
# $$
# 
# where $\textbf{w} = [w_0, w_1, w_2]^T$ is the vector of model parameters that the machine has to learn. You will soon realize that "learn" means solving an optimization problem to arrive to a set of optimal parameters. In this case, we could for example minimize the sum of squarred errors to get the optimal parameters $\textbf{w}^* $
# 
# $$
# \textbf{w}^* = argmin_{\textbf{w}} ~~ \frac{1}{2} \sum_{n=1}^N \left( y^{(n)} - h(\textbf{x}^{(n)}, \textbf{w}) \right)^2
# $$
# 
# This turns out to be a convex problem. This means that there exist only one optimum which is the global optimum. 
# 
# ```{attention}
# Did you remember the difference between local and global optima? 
# ```
# 
# There are many ways in which this optimization problem can be solved. For example, we can use gradient-based methods (e.g., [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)), Hessian-based methods (e.g., [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)) or, in this case, even an [analytical solution](https://math.stackexchange.com/questions/4177039/deriving-the-normal-equation-for-linear-regression) exists!

# ### What about non-linear problems? 🤔
# 
# We can expand this concept to cover non-linear spaces by introducing the concept of basis functions $\phi(\textbf{x})$ that map our original inputs to a different space where the problem becomes linear. Then, in this new space we perform linear regression (or classification) and effectively we are performing non-linear regression (classification) in the original space! Nice trick right?
# 
# This get rise to what we call Generalized Linear Models (GLM)!

# ## Linear regression 📉

# Let's now play around with this concepts by looking at a specific example: regressing thermophysical data of saturated and superheated vapor.
# 
# The data is taken from the Appendix F of {cite}`smith2004introduction`.
# 
# Let's import some libraries.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


# and then import the data

# In[2]:


if 'google.colab' in str(get_ipython()):
  df = pd.read_csv("https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/superheated_vapor.csv")
else:
  df = pd.read_csv("references/superheated_vapor.csv")

df.head()


# Many things to notice from observing the data above:
# 
# * We have 4 different properties V, U, H and S referring to the specific properties volume [cm$^3$ g$^{-1}$], internal energy [kJ kg$^{-1}$], enthalpy [kJ kg$^{-1}$] and entropy[kJ kg$^{-1}$ K$^{-1}$], respectively.
# * We have values for each property at different pressures [kPa] and temperatures [°C].
# * We have also the value of each property at each pressure for the saturated liquid and vapor.
# 
# Since we plan to build models per each property individually, let's separate the data per property.

# In[3]:


V = df.loc[df['Property'] == 'V']
U = df.loc[df['Property'] == 'U']
H = df.loc[df['Property'] == 'H']
S = df.loc[df['Property'] == 'S']


# Ploting the data, whenever possible, is always useful! So, let's plot for instance the saturated liquid data to see what trends it follows

# In[4]:


property_to_plot = 'Liq_Sat'

# Plot saturated liquid
plt.figure(figsize=(13, 7))
plt.subplot(221)
plt.plot(V['Pressure'], V[property_to_plot], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^3 g^{-1}$]')

plt.subplot(222)
plt.plot(U['Pressure'], U[property_to_plot], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific internal energy [$kJ kg^{-1}$]')

plt.subplot(223)
plt.plot(H['Pressure'], H[property_to_plot], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific enthalpy [$kJ kg^{-1}$]')

plt.subplot(224)
plt.plot(S['Pressure'], S[property_to_plot], 'kx', markersize=3)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific enthropy [$kJ kg^{-1} K^{-1}$]')

plt.suptitle(property_to_plot, size=15)
plt.show()


# We can also get some statical description of the data really fast by calling the method [`describe()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) on a DataFrame or Series

# In[5]:


V['Liq_Sat'].describe()


# Well, the trend is clearly non-linear in the whole range of pressures right?
# 
# We will focus in this exercise on a single property, let's take for instance the specific volume of the saturated liquid.
# 
# Our goal is to build a mathematical model (for now using (generalized) linear regression), that predicts the specific volume of a saturated liquid as a function of the pressure. This means that we are dealing with a one-dimensional problem.
# 
# It is also quite noticible to see that a simple line is not going to fit the data very well on the whole range of pressures. Do you see why looking at your data first is important? This process is formally known as [Exploratory Data Analysis (EDA)](https://en.wikipedia.org/wiki/Exploratory_data_analysis) and refers to the general task of analyzing the data using statistical metrics, visualization tools, etc...
# 
# ```{important}
# Before starting modeling, explore your data!
# ```
# 
# For now, let's assume we only know how to do a simple linear regression. Maybe one idea would be to split the data into sections and approximate each section with a different linear regression. This of course will result in a discrete model that is not very handdy... But let's try

# In[6]:


lim_1 = 300
lim_2 = 1500

plt.figure()
plt.plot(V['Pressure'], V['Liq_Sat'], 'kx', markersize=3)
plt.axvline(lim_1, linestyle='--', color='r')
plt.axvline(lim_2, linestyle='--', color='r')
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^3 g^{-1}$]')
plt.title('Saturated liquid', size=15)
plt.show()


#  Let's extract these sections from our data

# In[7]:


First_P  = V['Pressure'].loc[V['Pressure'] < lim_1].to_numpy().reshape(-1, 1)
First_V  = V['Liq_Sat'].loc[V['Pressure'] < lim_1].to_numpy().reshape(-1, 1)

Second_P = V['Pressure'].loc[(V['Pressure'] >= lim_1) & 
                                  (V['Pressure'] < lim_2)].to_numpy().reshape(-1, 1)
Second_V = V['Liq_Sat'].loc[(V['Pressure'] >= lim_1) & 
                                  (V['Pressure'] < lim_2)].to_numpy().reshape(-1, 1)

Third_P  = V['Pressure'].loc[V['Pressure'] >= lim_2].to_numpy().reshape(-1, 1)
Third_V  = V['Liq_Sat'].loc[V['Pressure'] >= lim_2].to_numpy().reshape(-1, 1)


# Now, we fit the 3 models

# In[8]:


LR_1 = LinearRegression().fit(First_P, First_V)
LR_2 = LinearRegression().fit(Second_P, Second_V)
LR_3 = LinearRegression().fit(Third_P, Third_V)

# - Plot splits and models
plt.figure()
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^3 g^{-1}$]')

# -- First split
plt.plot(First_P, First_V, 'rx', markersize=2, label='1st Section')    
plt.plot(np.linspace(0, lim_1 + 1000, 100), 
         LR_1.predict(np.linspace(0,lim_1 + 1000, 100).reshape(-1, 1)),'r', linewidth=1) 

# -- Second split
plt.plot(Second_P, Second_V, 'bx', markersize=2, label='2nd Section')  
plt.plot(np.linspace(lim_1, lim_2+1000, 100), 
         LR_2.predict(np.linspace(lim_1, lim_2+1000, 100).reshape(-1, 1)),'b', linewidth=1)


# -- Third split
plt.plot(Third_P, Third_V, 'kx', markersize=2, label='3rd Section')    
plt.plot(np.linspace(lim_2, max(Third_P), 100), 
         LR_3.predict(np.linspace(lim_2, max(Third_P), 100).reshape(-1, 1)),'k', linewidth=1)

plt.legend()
plt.show()


# We can check how well our model is fitting our data by printing the R$^2$ coefficient
# 
# The coefficient R$^2$ is defined as $1 - \frac{u}{v}$, where $u$ is the residual sum of squares $\sum (y - \hat{y})^2$ and $v$ is the total sum of squares $\sum (y - \mu_y)^2$. Here, $y$ denotes the true output value, $\hat{y}$ denotes the predicted output and $\mu_y$ stands for the mean of the output data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the mean of $y$, disregarding the input features, would get a R$^2$ score of 0.0.

# In[9]:


print('\n R2 for 1st split:', LR_1.score(First_P, First_V))
print('\n R2 for 2nd split:', LR_2.score(Second_P, Second_V))
print('\n R2 for 3rd split:', LR_3.score(Third_P, Third_V))


# ```{admonition} Question
# :class: hint
# What about training and test split? 😟 Can my model extrapolate to other pressures accurately? 
# ```

# To check the model parameters $\textbf{w}$

# In[10]:


print('\n Slope for 1st split    :', LR_1.coef_)
print(' Intercept for 1st split:', LR_1.intercept_)

print('\n Slope for 2nd split    :', LR_2.coef_)
print(' Intercept for 2nd split:', LR_2.intercept_)

print('\n Slope for 3rd split    :', LR_3.coef_)
print(' Intercept for 3rd split:', LR_3.intercept_)


# ```{admonition} Question
# :class: hint
# What about scaling? 😟 Does scaling matters for linear regression? Why or why not?
# ```

# ## Multivariate-linear regression 🤹
# 
# In our data set, we have a bunch of data corresponding to superheated vapor. We can take a look into it to see how can we create a mathematical model for it. This would be useful later for process simulation or process optimization. 
# 
# Let's pick just one property for now, the enthalpy.

# In[11]:


#%matplotlib qt #Uncomment this line to plot in a separate window (not available in Colab)

fig = plt.figure(figsize=(10,5))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

x = H['Pressure']
y = np.array([int(H.columns[4:][i]) for i in range(len(H.columns[4:]))])
X,Y = np.meshgrid(x,y)
Z = H.loc[:, '75':'650']

ax.scatter(X, Y, Z.T)
ax.set_xlabel('Pressure [kPa]')
ax.set_ylabel('Temperature [°C]')
ax.set_zlabel('Specific enthalpy [$kJ kg^{-1}$]')
ax.set_title('Superheated vapor', size=15)
plt.show()


# It seems like a multivariate linear regression would fit our experimental data quite well, right?
# 
# You will notice that the enthalpy data contains NaN values (i.e., empty values), which we should remove before fitting our model

# In[12]:


Ps   = X.reshape(-1,1)
Ts   = Y.reshape(-1,1)
Hs   = np.array(Z.T).reshape(-1,1)

# -- Clean data to eliminate NaN which cannot be used to fit the LR
H_bool  = np.isnan(Hs)
P_clean = np.zeros(len(Ps)-np.count_nonzero(H_bool))
T_clean = np.zeros(len(P_clean))
H_clean = np.zeros(len(P_clean))

j = 0
for i in range(Ps.shape[0]):
    if H_bool[i] == False:
        P_clean[j] = Ps[i]
        T_clean[j] = Ts[i]
        H_clean[j] = Hs[i]
        j += 1


# ### Exercise - multi-variate linear regression ❗❗
# 
# * Fit a multi-variate linear regression to the enthalpy data
# * What are the model parameters?
# * What is the R$^2$?
# * Plot your model using [`plot_surface`](https://matplotlib.org/stable/gallery/mplot3d/surface3d.html) along with the experimental data

# In[13]:


# Your code for the linear regression here


# In[14]:


# Your code for model parameters here


# In[15]:


# Your code for R2 here


# In[16]:


# Your code for plotting here


# ## Generalized Linear Regression 📈

# Previously we sectioned the volume data for a saturated liquid into three parts and we approximate each part with a
# linear regression. In this way, we obtained a discrete model for the whole range of pressures. This
# type of discrete models can be sometimes troublesome when applied to optimization problems (e.g.
# they have points where the gradient does not exist). A non-linear smooth model would be preferable (e.g., some sort of polynomial).

# In[17]:


Ps = V['Pressure'].to_numpy().reshape(-1,1)
Vs = V['Liq_Sat'].to_numpy().reshape(-1,1)


# Let's create some polynomial basis functions. For instance:
# 
# * $\phi_1(x) = x$
# * $\phi_2(x) = x^2$
# * $\phi_3(x) = x^3$
# * $\phi_4(x) = x^4$
# 
# Notice, that by using these basis functions we will be moving the problem from one-dimension to four-dimensions, in which the new input features are given by the corresponding basis function. We will integrate this series of transformations into a [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) just to exemplify its use.

# In[18]:


pf = PolynomialFeatures(degree=4, include_bias=False)
LR = LinearRegression()

# Define pipeline
GLM = Pipeline([("pf", pf),  ("LR", LR) ])
GLM.fit(Ps, Vs)


# A pipeline helps us in sequentially applying a list of transformations to the data. This could help us in preventing that we forget to apply one of the transformations when using our model.
# 
# Let's plot now the GLM

# In[19]:


evaluation_points = np.linspace(Ps[0], Ps[-1], 100)

plt.figure()
plt.plot(Ps, Vs, 'kx', markersize=3)
plt.plot(evaluation_points, GLM.predict(evaluation_points), 'b', linewidth=1)
plt.title('Polynomial regression', size=15)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^3 g^{-1}$]')
plt.xlim((0, Ps[-1]))
plt.ylim((Vs[0], Vs[-1]))
plt.show()


# In[20]:


# Print model parameters
print('Parameters: ', GLM['LR'].coef_)
print('Intercept : ', GLM['LR'].intercept_)


# In[21]:


print('\n R2 for GLM:', GLM.score(Ps, Vs))


# ### Exercise - polynomial regression ❗❗
# 
# * Create a function that takes as inputs 1) the pressure data and 2) the model parameters and returns the specific volume.
# * Use the polynomial model parameters obtained with sklearn as 2)
# * Plot the predictions along with the experimental data. Do you obtain the same plot?

# In[22]:


def poly_regression(Ps, model_params):
    w_0 = model_params[0]
    w_1 = model_params[1]
    w_2 = model_params[2]
    w_3 = model_params[3]
    w_4 = model_params[4]
    
    # Your code here

# Your code for usign the function here


# In[23]:


# Your code for plotting here


# ### Using a different base

# Generalized Linear Models are not restriceted to polynomial features. We can, in principle, use any basis functions that we want. How can we use for example a logarithmic basis function?

# In[24]:


class log_feature(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
       X_log = np.log(X)
       X_new = np.hstack((X, X_log))
       return X_new


# In[25]:


log_f = log_feature()

GLM_log = Pipeline([("log_f", log_f), ("LR", LR),])
GLM_log.fit(Ps, Vs)


# In[26]:


evaluation_points = np.linspace(Ps[0], Ps[-1], 100)

plt.figure()
plt.plot(Ps, Vs, 'kx', markersize=3)
plt.plot(evaluation_points, GLM_log.predict(evaluation_points), 'b', linewidth=1)
plt.title('Generalized linear regression', size=15)
plt.xlabel('Pressure [kPa]')
plt.ylabel('Specific volume [$cm^3 g^{-1}$]')
plt.xlim((0, Ps[-1]))
plt.ylim((Vs[0], Vs[-1]))
plt.show()


# In[27]:


# Print model parameters
print('Parameters: ', GLM_log['LR'].coef_)
print('Intercept : ', GLM_log['LR'].intercept_)


# In[28]:


print('\n R2 for GLM:', GLM_log.score(Ps, Vs))


# Regarding physical interpretability, what are the advantages that you see when using the logarithmic basis function vs. the polynomials?
# 
# ```{hint} 
# Can the pressure become negative?
# ```

# ### Exercise - Generalized linear regression ❗❗
# 
# * Create a function that takes as inputs 1) the pressure data and 2) the model parameters and returns the specific volume.
# * Use the logaritmic model parameters obtained with sklearn as 2)
# * Plot the predictions along with the experimental data. Do you obtain the same plot?

# In[29]:


# Your code here


# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
