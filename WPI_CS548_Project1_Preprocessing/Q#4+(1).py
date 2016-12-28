
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
dataset = pd.read_csv(r"E:\CS 548\Project 1\Q#4_120.csv")

#Data = np.genfromtxt('E:\CS 548\Project 1\Q#4_120.csv', dtype=None, delimiter=',', skip_header=0, missing_values="NaN")

#data = np.delete(Data,(0),axis=0)
#imp = Imputer(missing_values='?', strategy='mean', axis=0)
#imp.fit(data)


# In[3]:

from pandas import DataFrame
import pandas as pd
Data = dataset.replace('?','0')
pca = PCA(n_components=120)
pca.fit(Data)
VarienceExplained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(VarienceExplained[2])
print(pca.components_[0])


# In[4]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
y = Data['murdPerPop']
measurements = pd.read_csv(r"E:\CS 548\Project 1\Q#4_120_measurement.csv")
X = measurements.replace('?','0')
attributes = pd.read_csv(r"E:\CS 548\Project 1\Q#4_120_attributes.csv")
cfs = SelectKBest(chi2,3)
y = y.astype(int)
cfs.fit(X,y)
feature = cfs.get_support()
attr = np.array(attributes[[0]])
for i in range(len(feature)):
    if feature[i]==True:
        print(attr[i])

