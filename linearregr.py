import numpy as np
import pandas as pd
import sys

#ML Programming Assignment 01 - Aman, Mariam, Zohaib

def get_ypred(x, weights):
    y_pred = x * weights
    y_pred = y_pred.sum(axis=1)
    return y_pred

def get_error(y_true, y_pred):
    error = y_true - y_pred
    return error

def linear_regression(filename, learning_rate, threshold):
    dataset = pd.read_csv (filename, header=None)
    dataset.columns = dataset.columns + 1
    dataset = dataset.rename({dataset.columns[-1]: 'true_value'}, axis=1)
    weights = [0] * len(dataset.columns)
    y_true = dataset['true_value']
    x = dataset.drop('true_value', axis=1)
    x[0] = 1
    x = x[[ind for ind in range(len(x.columns))]]
    result_df = pd.DataFrame(columns=[ind for ind in range(len(weights)+2)])
    itr = 0
    sse_diff = threshold + 0.1
    new_weights = []
    pre_sse = 10e100
    while sse_diff > threshold:
        y_pred = get_ypred(x, weights)
        err = get_error(y_true, y_pred)
        sse = (err * err).sum()
        sse_diff = pre_sse - sse
        pre_sse = sse
        for ind in range(len(weights)):
            wi = weights[ind]
            xi = x[ind]
            grad = (xi * err).sum()
            wi = wi + (learning_rate * grad)
            new_weights.append(wi)
        print([itr] + weights + [sse])
        result_df = result_df.append(pd.DataFrame([itr] + weights + [sse]).T, ignore_index = True)
        weights = new_weights
        new_weights = []
        itr = itr + 1

    return result_df


if __name__ == '__main__':
    arguments_passed = sys.argv
    properties_dict = {}
    for ind in range(len(arguments_passed)):
        if ind == 0:
            continue
        if '--' in arguments_passed[ind]:
            key = arguments_passed[ind].split('--')[1]
            val = arguments_passed[ind + 1]
            if ind + 1 == 4 or ind + 1 == 6:
                val = float(val)
            properties_dict[key] = val

    result = linear_regression(properties_dict['data'], properties_dict['learningRate'], properties_dict['threshold'])
    result = result.round(4)
    result.to_csv('result_' + properties_dict['data'], header=None, index=None)

