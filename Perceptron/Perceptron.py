#Aurthor : Mariam Riaz
# Perceptron using using fixed and annealing learning rates
import codecs
import numpy as np
import pandas as pd
import sys


def get_ypred(x, weights):
    y_pred = x * weights
    y_pred = y_pred.sum(axis=1)
    y_pred = y_pred.apply(lambda x: 1 if x>0 else 0)
    return y_pred

def get_error(y_true, y_pred):
    error = y_true - y_pred
    return error

def perceptron_FixLearningRate(data_frame, output_file):
    data_frame = pd.read_table(data_frame, header=None)
    data_frame = data_frame.dropna(axis=1, how="all")
    data_frame = data_frame.rename({data_frame.columns[0]: 'true_value'}, axis=1)
    data_frame['true_value'].replace('A', '1', inplace=True)
    data_frame['true_value'].replace('B', '0', inplace=True)
    data_frame['true_value'] = data_frame['true_value'].astype('int')
    weights = [0] * len(data_frame.columns)
    y_true = data_frame['true_value']
    x =data_frame.drop('true_value', axis=1)
    x[0] = 1
    x = x[[ind for ind in range(len(x.columns))]]
    itr = 0
    learning_rate = 1
    new_weights = []

    while itr < 101:
        y_pred = get_ypred(x, weights)
        err = get_error(y_true, y_pred)
        error_rate = y_true == y_pred
        error_rate = error_rate.value_counts()[False]
        for ind in range(len(weights)):
            wi = weights[ind]
            xi = x[ind]
            grad = (xi * err).sum()
            wi = wi + (learning_rate * grad)
            new_weights.append(wi)
        output_file.write(str(error_rate) + '\t')
        weights = new_weights
        new_weights = []
        itr = itr + 1

def perceptron_AnnealingLearningRate(data_frame, output_file):
    data_frame = pd.read_table(data_frame, header=None)
    data_frame = data_frame.dropna(axis=1, how="all")
    data_frame = data_frame.rename({data_frame.columns[0]: 'true_value'}, axis=1)
    data_frame['true_value'].replace('A', '1', inplace=True)
    data_frame['true_value'].replace('B', '0', inplace=True)
    data_frame['true_value'] = data_frame['true_value'].astype('int')
    weights = [0] * len(data_frame.columns)
    y_true = data_frame['true_value']
    x =data_frame.drop('true_value', axis=1)
    x[0] = 1
    x = x[[ind for ind in range(len(x.columns))]]
    itr = 0
    learning_rate = 1
    new_weights = []

    while itr < 101:
        y_pred = get_ypred(x, weights)
        err = get_error(y_true, y_pred)
        error_rate = y_true == y_pred
        error_rate = error_rate.value_counts()[False]
        for ind in range(len(weights)):
            wi = weights[ind]
            xi = x[ind]
            grad = (xi * err).sum()
            wi = wi + (learning_rate * grad)
            new_weights.append(wi)
        output_file.write(str(error_rate) + '\t')
        weights = new_weights
        new_weights = []
        itr = itr + 1
        learning_rate = 1 /(itr+1)


if __name__ == '__main__':
    arguments_passed = sys.argv
    properties_dict = {}
    for ind in range(len(arguments_passed)):
        if ind == 0:
            continue
        if '--' in arguments_passed[ind]:
            key = arguments_passed[ind].split('--')[1]
            val = arguments_passed[ind + 1]
            properties_dict[key] = val

result_file = codecs.open(properties_dict['output'], 'w', "utf-8")
result = perceptron_FixLearningRate(properties_dict['data'], result_file)
result_file.write('\n')
result_two = perceptron_AnnealingLearningRate(properties_dict['data'], result_file)
result_file.close()

