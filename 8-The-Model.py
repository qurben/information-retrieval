#!/usr/bin/env python
# coding: utf-8

# # The Model
# 
# Train a LambdaMART model using xgboost

# In[ ]:


import numpy as np
import xgboost as xgb
import pandas as pd
import math
import matplotlib

TEST_FILE = "./test_features.txt"
TRAIN_FILE = "./training_features.txt"
VALIDATE_FILE = "./validation_features.txt"

TRAINED_MODEL = "trained.model"


# In[ ]:


def rr(data):
    if sum(data)==0:
        return 0
    else:
        try:
            index = data.index(1, 0, 8)
            return 1/(index+1)
        except ValueError:
            return 0

def averagePrecision(data):
    if sum(data)==0:
        return 0
    total = len(data)
    return data[data == 1]/total


# Load the datasets used for training

# In[ ]:


dtrain = xgb.DMatrix(TRAIN_FILE)
dvalidation = xgb.DMatrix(VALIDATE_FILE)

params = {
    'objective' : 'rank:pairwise',
    'eval_metric': ['map'],
}

model =  xgb.train(params, dtrain, num_boost_round=300, evals=[(dvalidation, 'validation')], early_stopping_rounds=30, verbose_eval=True)
model.save_model(TRAINED_MODEL)


# Restore the model from file

# In[ ]:


dtest = xgb.DMatrix(TEST_FILE)

booster = xgb.Booster()
booster.load_model(TRAINED_MODEL)


# Calculate the results

# In[ ]:


preds = booster.predict(dtest, ntree_limit= model.best_ntree_limit)
print(preds)

