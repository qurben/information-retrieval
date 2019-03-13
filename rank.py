import numpy as np
import xgboost as xgb
import pandas as pd
import math
import matplotlib

# Train and return model
def train(dtrain, dvalidation):
    # dtrain = processFeatures(train)
    # dvalidation = processFeatures(validation)

    params = {
        'objective' : 'rank:pairwise',
        'eval_metric': ['map'],
    }

    
    return xgb.train(params, dtrain, num_boost_round=3, evals=[(dvalidation, 'validation')], early_stopping_rounds=30, verbose_eval=True)

# Return prediction based on the best tree
def rank(test, model):
    return model.predict(test, ntree_limit= model.best_ntree_limit)

# Convert from pandas to DMatrix with groups.
# Groups are how many pairs there are (so should be ordered by queries, and count how many identicals there are).
# Remve queries from actual features afterwards
def processFeatures(candidates):
    x = 1
    previousLabel = ""
    group = []
    for index, row in candidates.iterrows():
        if index != 0:
            if previousLabel != row['Query']:
                group.append(x)
                x  = 1
                continue

            x += 1

        previousLabel = row["Query"]

    group.append(x)
    dmatrix = xgb.DMatrix(candidates.drop("Query", axis = 1), candidates["Query"])
    dmatrix.set_group(group)
    return dmatrix

# labels1 = pd.DataFrame(np.random.randint(2, size=500), columns = ["Query"])
# train_data = pd.DataFrame(data = np.random.rand(500, 5))
# validate_data = pd.DataFrame(data = np.random.rand(500, 5))
# train_data["Query"] =  labels1
# validate_data["Query"] = labels1


train_data = xgb.DMatrix('../../../Downloads/Fold1/train.txt')
validate_data = xgb.DMatrix('../../../Downloads/Fold1/vali.txt')

model = train(train_data, validate_data)


# test_d =  np.random.rand(1, 5)
# test_data = pd.DataFrame(data = test_d)
# matrix_test_data = xgb.DMatrix(test_data)

testing_data = xgb.DMatrix('../../../Downloads/Fold1/test.txt')
testing_labels = testing_data.get_label()

preds = rank(testing_data, model)
print(preds)

groups = [135, 140]


def rr(data):
    if sum(data)==0:
        return 0
    else:
        try:
            index = data.index(1, 0, 8)
            return 1/(index+1)
        except ValueError as identifier:
            return 0

def averagePrecision(data):
    total = len(data)
    return data[data == 1]/total

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
        print("data", ordered)
        rrs.append(rr(ordered))
        ps.append(averagePrecision(ordered))
        lower=upper

mrr = 0
if len(rrs) != 0:
    mrr = sum(rrs)/len(rrs)

MeanAP = 0
if len(ps) != 0:
    MeanAP = sum(ps)/len(ps)

print(mrr, MeanAP)
