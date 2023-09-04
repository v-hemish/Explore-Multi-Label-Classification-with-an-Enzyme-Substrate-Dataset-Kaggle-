#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas numpy scikit-learn xgboost')

from sklearn.model_selection import train_test_split, RandomizedSearchCV


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np


# In[6]:


train = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/train.csv')


# In[8]:


features = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n', 'EState_VSA1', 'EState_VSA2', 
            'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha', 'HeavyAtomMolWt',
            'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex', 'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 
            'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9', 'fr_COO', 'fr_COO2']


# In[10]:


targets = ['EC1', 'EC2']


# In[13]:


X = train[features]
Y1 = train[targets[0]]
Y2 = train[targets[1]]


# In[14]:


X_train, X_val, Y1_train, Y1_val = train_test_split(X, Y1, test_size=0.2, random_state=42)


# In[17]:


param_grid = {
    "n_estimators": [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
    "max_depth": [int(x) for x in np.linspace(3, 9, num = 7)],
    "learning_rate": [x for x in np.linspace(0.01, 0.5, num = 50)],
    "gamma": [x for x in np.linspace(0.0, 0.5, num = 50)],
    "colsample_bytree": [x for x in np.linspace(0.3, 1.0, num = 50)],
}


# In[18]:


model = XGBClassifier()


# In[19]:


random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, Y1_train)


# In[21]:


print("Best parameters found: ", random_search.best_params_)


# In[22]:


model = XGBClassifier(**random_search.best_params_)
model.fit(X_train, Y1_train)


# In[23]:


test = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/test.csv')


# In[24]:


test_features = test[features]

# Predict the probabilities for the test data
test_predictions = test[['id']].copy()  # Create a new dataframe starting with 'id'
test_predictions['EC1'] = model.predict_proba(test_features)[:,1]


# In[ ]:


X_train, X_val, Y2_train, Y2_val = train_test_split(X, Y2, test_size=0.2, random_state=42)
random_search.fit(X_train, Y2_train)


# In[ ]:


model = XGBClassifier(**random_search.best_params_)
model.fit(X_train, Y2_train)

# Predict the probabilities for the test data
test_predictions['EC2'] = model.predict_proba(test_features)[:,1]


# In[ ]:


test_predictions.to_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/submission.csv', index=False)


# In[ ]:





# In[ ]:




