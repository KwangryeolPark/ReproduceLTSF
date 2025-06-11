import numpy as np


def RSE(pred, true):
    true_mean = true.mean()
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - true_mean) ** 2)
    return np.sqrt(numerator) / (np.sqrt(denominator) + 1e-12)


def CORR(pred, true):
    true_mean = true.mean(0)
    pred_mean = pred.mean(0)
    u = np.sum((true - true_mean) * (pred - pred_mean), axis=0)
    d = np.sqrt(np.sum((true - true_mean) ** 2, axis=0) * np.sum((pred - pred_mean) ** 2, axis=0)) + 1e-12
    return 0.01 * np.mean(u / d)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    denom = np.abs(true) + 1e-8
    return np.mean(np.abs((pred - true) / denom))


def MSPE(pred, true):
    denom = np.square(true) + 1e-8
    return np.mean(np.square((pred - true) / denom))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr
