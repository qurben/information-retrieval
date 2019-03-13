#!/usr/bin/env python
# coding: utf-8

# # The Model
# 
# Train a LambdaMART model using xgboost

# In[ ]:


import pandas as pd
import xgboost as xgb

TRAINING = 'training_features.csv'
TEST = 'test_features.csv'
VALIDATION = 'validation_features.csv'


# In[ ]:


def to_matrix(candidates):
    candidates = candidates.drop('Prefix', axis=1)
    candidates = candidates.drop('Suffix', axis=1)
    dmatrix = xgb.DMatrix(candidates.drop('Query', axis=1), candidates['Query'])
    return dmatrix

def train(train, validation):
    dtrain = to_matrix(train)
    dvalidation = to_matrix(validation)

    params = {
        'objective' : 'rank:pairwise',
        'eval_metric': 'error',
    }

    return xgb.train(
        params, 
        dtrain, 
        num_boost_round=300, 
        evals=[(dvalidation, 'validation')], 
        early_stopping_rounds=30, 
        verbose_eval=True
    )

def rank(test, model):
    dmatrix = to_matrix(test)
    return model.predict(test, ntree_limit=model.best_ntree_limit)


# Load the datasets used for training

# In[ ]:


training_data = pd.read_csv(TRAINING, low_memory=False)
validation_data = pd.read_csv(VALIDATION, low_memory=False)


# Train the model using these sets

# In[ ]:


model = train(training_data, validation_data)


# Delete the datasets from the memory

# In[ ]:


del training_data
del validation_data


# Load the test data and verify the results

# In[ ]:


test_data = pd.read_csv(TEST, low_memory=False)


# In[ ]:


result = rank(test_data, model)

