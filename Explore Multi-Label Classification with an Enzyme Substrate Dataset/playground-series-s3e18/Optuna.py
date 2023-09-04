#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas xgboost scikit-learn optuna')


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


# In[5]:


train = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/train.csv')


# In[7]:


features = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n', 'EState_VSA1', 'EState_VSA2', 
            'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha', 'HeavyAtomMolWt',
            'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex', 'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 
            'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9', 'fr_COO', 'fr_COO2']


# In[9]:


targets = ['EC1', 'EC2']


# In[12]:


X = train[features]
Y1 = train[targets[0]]
Y2 = train[targets[1]]


# In[14]:


X_train, X_val, Y1_train, Y1_val = train_test_split(X, Y1, test_size=0.2, random_state=42)


# In[16]:


def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.5),
        "gamma": trial.suggest_uniform("gamma", 0.0, 0.5),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
    }

    model = XGBClassifier(**param)
    model.fit(X_train, Y1_train)
    Y1_pred = model.predict(X_val)
    accuracy = accuracy_score(Y1_val, Y1_pred)
    return accuracy


# In[18]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)


# In[20]:


print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial


# In[22]:


print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[24]:


model = XGBClassifier(**study.best_params)
model.fit(X_train, Y1_train)


# In[26]:


test = pd.read_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/test.csv')
test_features = test[features]

# Predict the probabilities for the test data
test_predictions = test[['id']].copy()  # Create a new dataframe starting with 'id'
test_predictions['EC1'] = model.predict_proba(test_features)[:,1]


# In[28]:


test_predictions = test[['id']].copy()  # Create a new dataframe starting with 'id'
test_predictions['EC1'] = model.predict_proba(test_features)[:,1]


# In[30]:


X_train, X_val, Y2_train, Y2_val = train_test_split(X, Y2, test_size=0.2, random_state=42)
study.optimize(objective, n_trials=100)


# In[32]:


model = XGBClassifier(**study.best_params)
model.fit(X_train, Y2_train)


# In[34]:


test_predictions['EC2'] = model.predict_proba(test_features)[:,1]


# In[35]:


test_predictions.to_csv('C:/Users/User/Desktop/Explore Multi-Label Classification with an Enzyme Substrate Dataset/playground-series-s3e18/submission.csv', index=False)

