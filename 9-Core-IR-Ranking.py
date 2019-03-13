#!/usr/bin/env python
# coding: utf-8

# # Ranking

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

# In[ ]:

dtrain = xgb.DMatrix(TRAIN_FILE)
dvalidation = xgb.DMatrix(VALIDATE_FILE)

params = {
    'objective' : 'rank:pairwise',
    'eval_metric': ['map'],
}

model =  xgb.train(params, dtrain, num_boost_round=300, evals=[(dvalidation, 'validation')], early_stopping_rounds=30, verbose_eval=True)
model.save_model(TRAINED_MODEL)


# In[ ]:

dtest = xgb.DMatrix(TEST_FILE)

booster = xgb.Booster()
booster.load_model(TRAINED_MODEL)

# def rank(test, model):
preds = booster.predict(dtest, ntree_limit= model.best_ntree_limit)
print(preds)


# In[ ]:

testing_labels = dtest.get_label()


groups = []

fi = open(TEST_FILE)
size = 1

group = ""
for line in fi:
    if not line:
        break
    splits = line.strip().split(" ")
    if splits[1] != group:
        group = splits[1]
        groups.append(size)
        size = 1

    size += 1

# print(group)
# print(groups)

nquerys=range(0,len(groups))
lower=0
upper=0
rrs=[]
ps = []
for i in range(0, len( groups)):
        many=groups[i]
        upper = upper+many
        predicted = preds[lower:upper]
        # print("predicted", predicted)
        labled = testing_labels[lower:upper]
        # print("Labled", labled)
        ordered = [x for _,x in sorted(zip(predicted,labled), reverse=True)][:8]
        # print("data", ordered)
        rrs.append(rr(ordered))
        ps.append(averagePrecision(ordered))
        lower=upper

mrr = 0
if len(rrs) != 0:
    mrr = sum(rrs)/len(rrs)

MeanAP = 0
if len(ps) != 0:
    MeanAP = sum(ps)/len(ps)

print("mrr:", mrr)
print("map:", MeanAP)


