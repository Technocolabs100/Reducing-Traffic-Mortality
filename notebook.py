
# coding: utf-8

# ## 1. The raw data files and their format
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_462/img/car-accident.jpg" alt=""></p>
# <p>While the rate of fatal road accidents has been decreasing steadily since the 80s, the past ten years have seen a stagnation in this reduction. Coupled with the increase in number of miles driven in the nation, the total number of traffic related-fatalities has now reached a ten year high and is rapidly increasing.</p>
# <p>Per request of the US Department of Transportation, we are currently investigating how to derive a strategy to reduce the incidence of road accidents across the nation. By looking at the demographics of traﬃc accident victims for each US state, we find that there is a lot of variation between states. Now we want to understand if there are patterns in this variation in order to derive suggestions for a policy action plan. In particular, instead of implementing a costly nation-wide plan we want to focus on groups of  states with similar profiles. How can we find such groups in a statistically sound way and communicate the result effectively?</p>
# <p>To accomplish these tasks, we will make use of data wrangling, plotting, dimensionality reduction, and unsupervised clustering.</p>
# <p>The data given to us was originally collected by the National Highway Traffic Safety Administration and the National Association of Insurance Commissioners. This particular dataset was compiled and released as a <a href="https://github.com/fivethirtyeight/data/tree/master/bad-drivers">CSV-file</a> by FiveThirtyEight under the <a href="https://github.com/ﬁvethirtyeight/data">CC-BY4.0 license</a>.</p>

# In[293]:


# Check the name of the current folder
current_dir = get_ipython().getoutput('pwd')
print(current_dir)

# List all files in this folder
file_list = get_ipython().getoutput('ls')
print(file_list)

# List all files in the datasets directory
dataset_list = get_ipython().getoutput('ls /datasets')
print(dataset_list)

# View the first 20 lines of datasets/road-accidents.csv
accidents_head = get_ipython().getoutput('head -n 20 datasets/road-accidents.csv')
accidents_head


# In[294]:


get_ipython().run_cell_magic('nose', '', "\nfrom pathlib import Path\n\n\ndef test_current_dir():\n    assert current_dir == [str(Path.cwd())], \\\n    'The current_dir variable was not correctly assigned.'\n    \n    \ndef test_file_list():\n    assert sorted(file_list) == sorted([str(p) for p in list(Path('.').glob('[A-z]*'))]), \\\n    'The file_list variable was not correctly assigned.'\n    \n    \ndef test_accidents_head():\n    with open('datasets/road-accidents.csv') as f:\n        accidents_head_test = []\n        for i in range(20):\n            accidents_head_test.append(f.readline().rstrip())\n    assert accidents_head == accidents_head_test, \\\n    'The accidents_head variable was not correctly assigned.'")


# ## 2. Read in and get an overview of the data
# <p>Next, we will orient ourselves to get to know the data with which we are dealing.</p>

# In[295]:


# Import the `pandas` module as "pd"
import pandas as pd

# Read in `road-accidents.csv`
car_acc = pd.read_csv('datasets/road-accidents.csv', comment='#', sep='|')

# Save the number of rows columns as a tuple
rows_and_cols = car_acc.shape
print('There are {} rows and {} columns.\n'.format(
    rows_and_cols[0], rows_and_cols[1]))

# Generate an overview of the DataFrame
car_acc_information = car_acc.info()
print(car_acc_information)

# Display the last five rows of the DataFrame
car_acc.tail()


# In[296]:


get_ipython().run_cell_magic('nose', '', '\nimport sys\n\ndef test_pandas_import():\n    assert \'pandas\' in list(sys.modules.keys()), \\\n        \'The pandas module has not been imported correctly.\'\n    \n\ndef test_car_acc():\n    car_acc_test = pd.read_csv(\'datasets/road-accidents.csv\', comment=\'#\', sep=\'|\')\n    try:\n        pd.testing.assert_frame_equal(car_acc, car_acc_test)\n    except AssertionError:\n        assert False, "The car_acc dataset was not read in correctly."\n        \n        \ndef test_car_acc_shape():\n    assert rows_and_cols == (51, 5), \\\n    \'The number of rows and variables were not calculated correctly.\'\n\n    \ndef test_car_acc_info():\n    assert car_acc_information == car_acc.info(), \\\n    \'The overview does not appear to be have created properly using the info method.\'')


# ## 3. Create a textual and a graphical summary of the data
# <p>We now have an idea of what the dataset looks like. To further familiarize ourselves with this data, we will calculate summary statistics and produce a graphical overview of the data. The graphical overview is good to get a sense for the distribution of variables within the data and could consist of one histogram per column. It is often a good idea to also explore the pairwise relationship between all columns in the data set by using a using pairwise scatter plots (sometimes referred to as a "scatterplot matrix").</p>

# In[297]:


# import seaborn and make plots appear inline
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Compute the summary statistics of all columns in the `car_acc` DataFrame
sum_stat_car = car_acc.describe()
print(sum_stat_car)

# Create a pairwise scatter plot to explore the data
sns.pairplot(car_acc)


# In[298]:


get_ipython().run_cell_magic('nose', '', '\nlast_value = _\n\nimport sys\n\ndef test_seaborn_import():\n    assert \'seaborn\' in list(sys.modules.keys()), \\\n        \'The seaborn module has not been imported correctly.\'\n    \n\ndef test_car_desc():\n    try:\n        pd.testing.assert_frame_equal(sum_stat_car, car_acc.describe())\n    except AssertionError:\n        assert False, "The sum_stat_car variable was not created correctly."\n\n\ndef test_pairplot_created():\n    assert type(last_value) == sns.axisgrid.PairGrid, \\\n    \'It does not appear that a Seaborn pairplot was the last output of the cell.\'')


# ## 4. Quantify the association of features and accidents
# <p>We can already see some potentially interesting relationships between the target variable (the number of fatal accidents) and the feature variables (the remaining three columns).</p>
# <p>To quantify the pairwise relationships that we observed in the scatter plots, we can compute the Pearson correlation coefficient matrix. The Pearson correlation coefficient is one of the most common methods to quantify correlation between variables, and by convention, the following thresholds are usually used:</p>
# <ul>
# <li>0.2 = weak</li>
# <li>0.5 = medium</li>
# <li>0.8 = strong</li>
# <li>0.9 = very strong</li>
# </ul>

# In[299]:


# Compute the correlation coefficent for all column pairs
corr_columns = car_acc.corr()
corr_columns


# In[300]:


get_ipython().run_cell_magic('nose', '', '\ndef test_corr_columns():\n    try:\n        pd.testing.assert_frame_equal(corr_columns, car_acc.corr())\n    except AssertionError:\n        assert False, "The corr_columns variable was not created correctly."')


# ## 5. Fit a multivariate linear regression
# <p>From the correlation table, we see that the amount of fatal accidents is most strongly correlated with alcohol consumption (first row). But in addition, we also see that some of the features are correlated with each other, for instance, speeding and alcohol consumption are positively correlated. We, therefore, want to compute the association of the target with each feature while adjusting for the effect of the remaining features. This can be done using multivariate linear regression.</p>
# <p>Both the multivariate regression and the correlation measure how strongly the features are associated with the outcome (fatal accidents). When comparing the regression coefficients with the correlation coefficients, we will see that they are slightly different. The reason for this is that the multiple regression computes the association of a feature with an outcome, given the association with all other features, which is not accounted for when calculating the correlation coefficients.</p>
# <p>A particularly interesting case is when the correlation coefficient and the regression coefficient of the same feature have opposite signs. How can this be? For example, when a feature A is positively correlated with the outcome Y but also positively correlated with a different feature B that has a negative effect on Y, then the indirect correlation (A-&gt;B-&gt;Y) can overwhelm the direct correlation (A-&gt;Y). In such a case, the regression coefficient of feature A could be positive, while the correlation coefficient is negative. This is sometimes called a <em>masking</em> relationship. Let’s see if the multivariate regression can reveal such a phenomenon.</p>

# In[301]:


# Import the linear model function from sklearn
from sklearn import linear_model

# Create the features and target DataFrames
features = car_acc[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
target = car_acc['drvr_fatl_col_bmiles']

# Create a linear regression object
reg = linear_model.LinearRegression()

# Fit a multivariate linear regression model
reg.fit(features, target)

# Retrieve the regression coefficients
fit_coef = reg.coef_
fit_coef


# In[302]:


get_ipython().run_cell_magic('nose', '', '\nimport sys\nimport numpy\n\n\ndef test_sklearn_import():\n    assert \'sklearn\' in list(sys.modules.keys()), \\\n        \'The seaborn module has not been imported correctly.\'\n    \n    \ndef test_features_df():\n    try:\n        pd.testing.assert_frame_equal(features, car_acc[[\'perc_fatl_speed\', \'perc_fatl_alcohol\', \'perc_fatl_1st_time\']])\n    except AssertionError:\n        assert False, "The features DataFrame was not created correctly."\n\n        \ndef test_target_df():\n    try:\n        pd.testing.assert_frame_equal(target.to_frame(), car_acc[[\'drvr_fatl_col_bmiles\']])\n    except AssertionError:\n        assert False, "The target DataFrame variable was not created correctly."\n        \n        \ndef test_lin_reg():\n    assert reg.coef_.round(3).tolist() == [-0.042,  0.191,  0.025], \\\n     \'The linear regression coefficients are not correct.\'')


# ## 6. Perform PCA on standardized data
# <p>We have learned that alcohol consumption is weakly associated with the number of fatal accidents across states. This could lead us to conclude that alcohol consumption should be a focus for further investigations and maybe strategies should divide states into high versus low alcohol consumption in accidents. But there are also associations between  alcohol consumptions and the other two features, so it might be worth trying to split the states in a way that accounts for all three features.</p>
# <p>One way of clustering the data is to use PCA to visualize data in reduced dimensional space where we can try to pick up patterns by eye. PCA uses the absolute variance to calculate the overall variance explained for each principal component, so it is important that the features are on a similar scale (unless we would have a particular reason that one feature should be weighted more).</p>
# <p>We'll use the appropriate scaling function to standardize the features to be centered with mean 0 and scaled with standard deviation 1.</p>

# In[303]:


# Standardize and center the feature columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Import the PCA class function from sklearn
from sklearn.decomposition import PCA
pca = PCA()

# Fit the standardized data to the pca
pca.fit(features_scaled)

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


# In[304]:


get_ipython().run_cell_magic('nose', '', "\nimport sys\nimport numpy\n\n\ndef test_scaler():\n    assert scaler.fit_transform(features).round(3).tolist()[-1] == [1.077, 0.259, 0.185], \\\n        'The scaled features were not calculated properly.'\n\n    \ndef test_pca():\n    assert (pca.explained_variance_ratio_ == PCA().fit(features_scaled).explained_variance_ratio_).all(), \\\n        'The explained variance ratio for the PCA was not correctly calculated.'\n    \n    \ndef test_pc1_pc2():\n    assert two_first_comp_var_exp == PCA().fit(features_scaled).explained_variance_ratio_.cumsum()[1], \\\n        'The cumulative sum for the explained variance of the two first principal components was not correctly calculated.'")


# ## 7. Visualize the first two principal components
# <p>The first two principal components enable visualization of the data in two dimensions while capturing a high proportion of the variation (79%) from all three features: speeding, alcohol influence, and first-time accidents. This enables us to use our eyes to try to discern patterns in the data with the goal to find groups of similar states. Although clustering algorithms are becoming increasingly efficient, human pattern recognition is an easily accessible and very efficient method of assessing patterns in data.</p>
# <p>We will create a scatter plot of the first principle components and explore how the states cluster together in this visualization.</p>

# In[305]:


# Transform the scaled features using two principal components
pca = PCA(n_components = 2)
p_comps = pca.fit_transform(features_scaled)

# Extract the first and second component to use for the scatter plot
p_comp1 = p_comps[:,0]
p_comp2 = p_comps[:,1]

# Plot the first two principal components in a scatter plot
plt.scatter(p_comp1, p_comp2)


# In[306]:


get_ipython().run_cell_magic('nose', '', "\ndef test_pca_trans():\n    assert (p_comps == PCA(n_components=2).fit_transform(features_scaled)).all(), \\\n        'The PCA transformation was not performed correctly'\n    \n\ndef test_pca_comp1():\n    assert (p_comp1 == p_comps[:, 0]).all(), \\\n        'The first principal component was not assigned correctly.'\n    \n\ndef test_pca_comp2():\n    assert (p_comp2 == p_comps[:, 1]).all(), \\\n        'The second principal component was not assigned correctly.'")


# ## 8. Find clusters of similar states in the data
# <p>It was not entirely clear from the PCA scatter plot how many groups in which the states cluster. To assist with identifying a reasonable number of clusters, we can use KMeans clustering by creating a scree plot and finding the "elbow", which is an indication of when the addition of more clusters does not add much explanatory power.</p>

# In[307]:


# Import KMeans from sklearn
from sklearn.cluster import KMeans

# A loop will be used to plot the explanatory power for up to 10 KMeans clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(ks, inertias, marker='o')


# In[308]:


get_ipython().run_cell_magic('nose', '', "\ndef test_inertias():\n    test_ins = [153.0, 101.591, 72.293, 57.791, 46.106, 39.213, 32.99, 29.381, 25.985]\n    assert [round(inertia, 3) for inertia in inertias] ==  test_ins, \\\n        'The list of inertias was not properly constructed.'\n\n    \ndef test_km():\n    assert (km.labels_ == KMeans(n_clusters=k, random_state=8).fit(features_scaled).labels_).all(), \\\n        'The KMeans labels were not properly assigned.'")


# ## 9. KMeans to visualize clusters in the PCA scatter plot
# <p>Since there wasn't a clear elbow in the scree plot, assigning the states to either two or three clusters is a reasonable choice, and we will resume our analysis using three clusters. Let's see how the PCA scatter plot looks if we color the states according to the cluster to which they are assigned.</p>

# In[309]:


# Create a KMeans object with 3 clusters, use random_state=8 
km = KMeans(n_clusters = 3, random_state = 8)

# Fit the data to the `km` object
km.fit(features_scaled)

# Create a scatter plot of the first two principal components
# and color it according to the KMeans cluster assignment 
plt.scatter(p_comp1, p_comp2, c=km.labels_)


# In[310]:


get_ipython().run_cell_magic('nose', '', "\ndef test_km_labels():\n    assert (km.labels_ == KMeans(n_clusters=3, random_state=8).fit(features_scaled).labels_).all(), \\\n        'The KMeans labels were not properly assigned.'")


# ## 10. Visualize the feature differences between the clusters
# <p>Thus far, we have used both our visual interpretation of the data and the KMeans clustering algorithm to reveal patterns in the data, but what do these patterns mean?</p>
# <p>Remember that the information we have used to cluster the states into three distinct groups are the percentage of drivers speeding, under alcohol influence and that has not previously been involved in an accident. We used these clusters to visualize how the states group together when considering the first two principal components. This is good for us to understand structure in the data, but not always easy to understand, especially not if the findings are to be communicated to a non-specialist audience.</p>
# <p>A reasonable next step in our analysis is to explore how the three clusters are different in terms of the three features that we used for clustering. Instead of using the scaled features, we return to using the unscaled features to help us interpret the differences.</p>

# In[311]:


# Create a new column with the labels from the KMeans clustering
car_acc['cluster'] = km.labels_

# Reshape the DataFrame to the long format
melt_car = pd.melt(car_acc, id_vars='cluster', 
                   var_name='measurement', 
                   value_name='percent', 
                   value_vars=features)

# Create a violin plot splitting and coloring the results according to the km-clusters
sns.violinplot(x = melt_car['percent'], y = melt_car['measurement'], hue = melt_car['cluster'])


# In[312]:


get_ipython().run_cell_magic('nose', '', '\ndef test_melt():\n    test_melt = pd.melt(car_acc, id_vars=\'cluster\', var_name=\'measurement\', value_name=\'percent\',\n       value_vars=[\'perc_fatl_speed\', \'perc_fatl_alcohol\', \'perc_fatl_1st_time\'])\n    try:\n        pd.testing.assert_frame_equal(melt_car, test_melt)\n    except AssertionError:\n        assert False, "The melt_car DataFrame was not created correctly."')


# ## 11. Compute the number of accidents within each cluster
# <p>Now it is clear that different groups of states may require different interventions. Since resources and time are limited, it is useful to start off with an intervention in one of the three groups first. Which group would this be? To determine this, we will include data on how many miles are driven in each state, because this will help us to compute the total number of fatal accidents in each state. Data on miles driven is available in another tab-delimited text file. We will assign this new information to a column in the DataFrame and create a violin plot for how many total fatal traffic accidents there are within each state cluster.</p>

# In[313]:


# Read in the new dataset
miles_driven = pd.read_csv('datasets/miles-driven.csv', sep='|')

# Merge the `car_acc` DataFrame with the `miles_driven` DataFrame
car_acc_miles = car_acc.merge(miles_driven, on='state')

# Create a new column for the number of drivers involved in fatal accidents
car_acc_miles['num_drvr_fatl_col'] = (car_acc_miles['drvr_fatl_col_bmiles'] * car_acc_miles["million_miles_annually"])/1000

# Create a barplot of the total number of accidents per cluster
sns.barplot(x='cluster', y='num_drvr_fatl_col', data=car_acc_miles, estimator=sum, ci=None)

# Calculate the number of states in each cluster and their 'num_drvr_fatl_col' mean and sum.
count_mean_sum = car_acc_miles.groupby("cluster")["num_drvr_fatl_col"].agg(["count", "mean", "sum"])
count_mean_sum


# In[314]:


get_ipython().run_cell_magic('nose', '', '\ndef test_miles_driven():\n    try:\n        pd.testing.assert_frame_equal(miles_driven, pd.read_csv(\'datasets/miles-driven.csv\', sep=\'|\'))\n    except AssertionError:\n        assert False, \'The miles_driven DataFrame was not read in correctly.\'\n\n\ndef test_merge_dfs():\n    try:\n        pd.testing.assert_frame_equal(car_acc_miles.drop(columns=\'num_drvr_fatl_col\'), pd.merge(car_acc, miles_driven, on=\'state\'))\n    except AssertionError:\n        assert False, \'The two DataFrames were not merged correctly.\'\n        \n        \ndef test_new_column():\n    new_col_df_test = car_acc_miles[\'drvr_fatl_col_bmiles\'] * car_acc_miles[\'million_miles_annually\'] / 1000\n    new_col_df_test.name = \'num_drvr_fatl_col\'\n    try:\n        pd.testing.assert_series_equal(car_acc_miles[\'num_drvr_fatl_col\'], new_col_df_test)\n    except AssertionError:\n        assert False, \'The new column "num_drvr_fatl_col" was not computed correctly.\'\n        \n        \ndef test_agg():\n    count_mean_sum_test = car_acc_miles.groupby(\'cluster\')[\'num_drvr_fatl_col\'].agg([\'count\', \'mean\', \'sum\'])\n    try:\n        pd.testing.assert_frame_equal(count_mean_sum, count_mean_sum_test)\n    except AssertionError:\n        assert False, (\'The aggregation step was not performed correctly. \'\n                       \'Note that the order should be 1. "count", 2. "mean", and 3. "sum".\')')


# ## 12. Make a decision when there is no clear right choice
# <p>As we can see, there is no obvious correct choice regarding which cluster is the most important to focus on. Yet, we can still argue for a certain cluster and motivate this using our findings above. Which cluster do you think should be a focus for policy intervention and further investigation?</p>

# In[315]:


# Which cluster would you choose?
cluster_num = 2


# In[316]:


get_ipython().run_cell_magic('nose', '--nocapture', "\ndef test_cluster_choice():\n    assert cluster_num in range(3), \\\n    'cluster_num must be either 0, 1, or 2'\n    print('Well done! Note that there is no definite correct answer here and there are a few ways to justify each cluster choice:'\n          '\\n0 (Blue) = The lowest number of states and the highest number of people helped per state. Good for a focused pilot effort.'\n          '\\n2 (Green) = The highest number of people helped in total and the most states. Good if we can mobilize many resources right away.'\n          '\\n1 (Orange) = A good balance of the attributes from the two other clusters. This cluster also has the highest alcohol consumption'\n          '\\nwhich was the strongest correlated to fatal accidents.')")

