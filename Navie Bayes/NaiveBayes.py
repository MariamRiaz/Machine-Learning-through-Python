import pandas as pd
import math
import sys
import codecs

def propability_class (file_name, result_file):
    data_frame = pd.read_table(file_name, header=None, float_precision='round_trip')
    data_frame = data_frame.dropna(axis=1, how="all")
    data_frame = data_frame.rename({data_frame.columns[0]: 'true_value'}, axis=1)
    x = data_frame['true_value'].shape
    y = data_frame['true_value'].value_counts()
    t = y/x
    P_A = t.iloc[0]
    P_B = t.iloc[1]
    by_class = data_frame.groupby('true_value')
    class_A = by_class.get_group('A')
    class_B = by_class.get_group('B')
    m1_A, sd1_A, m2_A, sd2_A = mns_class(class_A)
    m1_B, sd1_B, m2_B, sd2_B = mns_class(class_B)
    prob_gaus(data_frame, m1_A, sd1_A, m2_A, sd2_A , m1_B, sd1_B, m2_B, sd2_B, P_A, P_B, result_file)

def mns_class(dataset):
    a = dataset['true_value'].shape
    dataset = dataset.drop('true_value', axis=1)
    b = dataset.sum(axis=0, skipna=True)
    m = b/a[0]
    m1 = m.iloc[0]
    m2 = m.iloc[1]
    c = dataset[1].sub(m.iloc[0])
    d = dataset[2].sub(m.iloc[1])
    e = ((c*c).sum(axis=0, skipna=True))
    f = ((d*d).sum(axis=0, skipna=True))
    g = a[0] - 1
    sd1 = e/g
    sd2 = f/g
    return m1, sd1, m2, sd2





def prob_gaus(dataset, m1_A, sd1_A, m2_A, sd2_A , m1_B, sd1_B, m2_B, sd2_B, P_A, P_B, result_file):
    misclassif = 0
    for ind in range(dataset.shape[0]):
        sample = dataset.iloc[ind]
        P1_A = (1 / math.sqrt(2 * math.pi * sd1_A)) * math.exp(-(((sample.loc[1] - m1_A) ** 2) / (2 * sd1_A)))
        P2_A = (1 / math.sqrt(2 * math.pi * sd2_A)) * math.exp(-(((sample.loc[2] - m2_A) ** 2) / (2 * sd2_A)))
        P1_B = (1 / math.sqrt(2 * math.pi * sd1_B)) * math.exp(-(((sample.loc[1] - m1_B) ** 2) / (2 * sd1_B)))
        P2_B = (1 / math.sqrt(2 * math.pi * sd2_B)) * math.exp(-(((sample.loc[2] - m2_B) ** 2) / (2 * sd2_B)))
        P_sample_A = P1_A * P2_A
        P_sample_B = P1_B * P2_B
        Class_sample_A = ((P_sample_A * P_A ) / ((P_sample_A *P_A) + (P_sample_B * P_B)))
        Class_sample_B = ((P_sample_B * P_B) / ((P_sample_A * P_A) + (P_sample_B * P_B)))
        if Class_sample_A > Class_sample_B:
            class_of_sample = 'A'
        else:
            class_of_sample = 'B'

        if class_of_sample != sample.loc['true_value']:
            misclassif = misclassif + 1
    result_file.write(str(m1_A) + '\t' + str(sd1_A) + '\t' +  str(m2_A) + '\t' +  str(sd2_A) + '\t' +  str(P_A))
    result_file.write('\n')
    result_file.write(str(m1_B) + '\t' + str(sd1_B) + '\t' + str(m2_B) + '\t' + str(sd2_B) + '\t' +  str(P_B))
    result_file.write('\n')
    result_file.write(str(misclassif))

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

file_name = properties_dict["data"]
result_file = codecs.open(properties_dict['output'], 'w', "utf-8")
propability_class(file_name, result_file)
result_file.close()

