#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install pandas numpy scikit-learn xgboost')


# In[47]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# In[28]:


train = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/train.csv')
test = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/test.csv')


# In[29]:


train.head()


# In[30]:


train_id = train['id']
test_id = test['id']


# In[31]:


features = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha', 'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex', 'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9', 'fr_COO', 'fr_COO2']
X = train[features]
Y1 = train['EC1']
Y2 = train['EC2']


# In[32]:


X_train, X_val, Y1_train, Y1_val, Y2_train, Y2_val = train_test_split(X, Y1, Y2, test_size=0.2, random_state=7)


# In[48]:


param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0.0, 0.1, 0.2]
}


# In[49]:


grid1 = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid, cv=3)
grid1.fit(X_train, Y1_train)

grid2 = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid, cv=3)
grid2.fit(X_train, Y2_train)


# In[50]:


model1 = grid1.best_estimator_
model2 = grid2.best_estimator_


# In[52]:


Y1_pred = model1.predict(X_val)
Y2_pred = model2.predict(X_val)


# In[53]:


accuracy1 = accuracy_score(Y1_val, Y1_pred)
accuracy2 = accuracy_score(Y2_val, Y2_pred)

print("Accuracy for EC1: %.2f%%" % (accuracy1 * 100.0))
print("Accuracy for EC2: %.2f%%" % (accuracy2 * 100.0))


# In[54]:


test_features = test[features]
test_predictions = test[['id']].copy()  # Create a new dataframe starting with 'id'
test_predictions['EC1'] = model1.predict_proba(test_features)[:,1]
test_predictions['EC2'] = model2.predict_proba(test_features)[:,1]


# In[56]:


test_predictions.to_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/submission.csv', index=False)


# In[37]:





# In[42]:





# In[44]:




