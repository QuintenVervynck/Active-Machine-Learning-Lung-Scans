import numpy as np


def uncertainty(model, X_pool, query_size):
    query_size = min(query_size, X_pool.shape[0])
    # get predictions on remaining training data
    p_pool = model.predict(X_pool)
    # get indexes of most uncertain predictions) (if the max is low, then the prediction is uncertain)
    idxs = np.max(p_pool, axis=1).argsort()[:query_size]
    return idxs


def margin(model, X_pool, query_size):
    query_size = min(query_size, X_pool.shape[0])
    # get predictions on remaining training data
    p_pool = model.predict(X_pool)
    # get indexes of predictions with smallest margin (difference) between first and second most likely
    p_pool.sort(axis=1)
    idxs = np.array([(xi[-1] - xi[-2]) for xi in p_pool]).argsort()[:query_size]
    return idxs


def entropy(model, X_pool, query_size):
    query_size = min(query_size, X_pool.shape[0])
    # get predictions on remaining training data
    p_pool = model.predict(X_pool)
    # get indexes of predictions with highest entropy (-Sum(y):P(y)*log2(P(y))) (last indices have high entr.)
    idxs = -np.sum(np.multiply(p_pool, np.log2(p_pool)), axis=1).argsort()[-query_size:]
    return idxs