import xgboost as xgb

def train(test, train, validation):

    params = {
        'objective' : 'rank:pairwise',
        'eval_metric': 'error',
    }

    return xgb.train(params, train, num_boost_round=300, evals=[(validation, 'validation')], early_stopping_rounds=30)

def rank(test, model):
    return model.predict(test)