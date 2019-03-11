import numpy as np
import xgboost as xgb
import pandas as pd

# Train and return model
def train(train, validation):
    dtrain = processFeatures(train)
    dvalidation = processFeatures(validation)

    params = {
        'objective' : 'rank:pairwise',
        'eval_metric': 'error',
    }

    return xgb.train(params, dtrain, num_boost_round=300, evals=[(dvalidation, 'validation')], early_stopping_rounds=30, verbose_eval=True)

# Return prediction based on the best tree
def rank(test, model):
    return model.predict(xgb.DMatrix(test), ntree_limit= model.best_ntree_limit)

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
# train_data = pd.DataFrame(data = np.random.rand(500, 10))
# validate_data = pd.DataFrame(data = np.random.rand(500, 10))
# train_data["Query"] =  labels1
# validate_data["Query"] = labels1

# model = train(train_data, validate_data)

# test_data = pd.DataFrame(data = np.random.rand(2, 10))

# print(rank(test_data, model))
