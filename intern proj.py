#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

# Read in `road-accidents.csv`
file1 = pd.read_csv('C:/Users/Lenovo/Downloads/road-accidents.csv', comment = '#', sep = '|')

# Save the number of rows columns as a tuple
rows_and_cols = file1.shape
print('There are {} rows and {} columns.\n'.format(
    rows_and_cols[0], rows_and_cols[1]))

# Generate an overview of the DataFrame
file1_information = file1.info()
print(file1_information)

# Display the last five rows of the DataFrame
file1.tail()


# In[5]:


import seaborn as sns
%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')

# Compute the summary statistics of all columns in the `file1` DataFrame
sum_file1 = file1.describe()
print(sum_file1)
# Create a pairwise scatter plot to explore the data
sns.pairplot(sum_file1)


# In[6]:


# Compute the correlation coefficent for all column pairs
file1_columns = file1.corr()
file1_columns


# In[7]:


from sklearn import linear_model

# Create the features and target DataFrames
fea = file1[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
tar= file1['drvr_fatl_col_bmiles']

# Create a linear regression object
reg = linear_model.LinearRegression()

# Fit a multivariate linear regression model
reg.fit(fea, tar)

# Retrieve the regression coefficients
fit_coef = reg.coef_
fit_coef


# In[8]:


import numpy as np

# Standardize and center the feature columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fea_scaled = scaler.fit_transform(fea)

# Import the PCA class function from sklearn
from sklearn.decomposition import PCA
pca = PCA()

# Fit the standardized data to the pca
pca.fit(fea_scaled)
# Plot the proportion of variance explained on the y-axis of the bar plot
import matplotlib.pyplot as plt
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])

# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_[0].cumsum()[0] + pca.explained_variance_ratio_[1].cumsum()[0]
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 5)))


# In[9]:


# Transform the scaled features using two principal components
pca = PCA(n_components = 2)
p_comps = pca.fit_transform(fea_scaled)

# Extract the first and second component to use for the scatter plot
p_comp1 = p_comps[:, 0]
p_comp2 = p_comps[:, 1]

# Plot the first two principal components in a scatter plot
plt.scatter(p_comp1, p_comp2)


# In[10]:


# Import KMeans from sklearn
from sklearn.cluster import KMeans

# A loop will be used to plot the explanatory power for up to 10 KMeans clusters
kst = range(1, 10)
inertias = []
for k in kst:
    # Initialize the KMeans object using the current number of clusters (k)
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit(fea_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(kst, inertias, marker='o')


# In[13]:


# Create a KMeans object with 3 clusters, use random_state=8 
km = KMeans(n_clusters = 3, random_state = 8)

# Fit the data to the `km` object
km.fit(fea_scaled)

# Create a scatter plot of the first two principal components
# and color it according to the KMeans cluster assignment 
plt.scatter(p_comps[:, 0], p_comps[:, 1], c = km.labels_)


# In[14]:


# Create a new column with the labels from the KMeans clustering
file1['cluster'] = km.labels_

# Reshape the DataFrame to the long format
melt_car = pd.melt(file1, id_vars = ['cluster'], var_name ='measurement', value_name = 'percent', 
                                                   value_vars =['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time'])

# Create a violin plot splitting and coloring the results according to the km-clusters
sns.violinplot(melt_car['percent'], melt_car['measurement'], hue = melt_car['cluster'])


# In[15]:


# Read in the new dataset
file2= pd.read_csv('C:/Users/Lenovo/Downloads/miles-driven.csv', sep='|')

display(file2.head())

# Merge the `car_acc` DataFrame with the `miles_driven` DataFrame
merge = file1.merge(file2, on='state')

# Create a new column for the number of drivers involved in fatal accidents
merge['num_drvr_fatl_col'] = (merge['drvr_fatl_col_bmiles'] * merge['million_miles_annually']) / 1000

display(merge.head())

# Create a barplot of the total number of accidents per cluster
sns.barplot(x='cluster', y='num_drvr_fatl_col', data=merge, estimator=sum, ci=None)

# Calculate the number of states in each cluster and their 'num_drvr_fatl_col' mean and sum.
count_mean_sum = merge.groupby('cluster')['num_drvr_fatl_col'].agg(['count', 'mean', 'sum'])
count_mean_sum


# In[16]:


# Which cluster would you choose?
cluster_num = ...


# In[ ]:




