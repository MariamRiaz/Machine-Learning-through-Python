#Aurthor : Mariam Riaz
# Decision tree using ID3 algorithm using information gain as the split function

import math
import pandas as pd
import sys
import xml.etree.cElementTree as ET


def entropy(df, nc): #Entropy function ; calculates entropy attribute value pair that is passed
    vp = df['true_value'].value_counts()
    pp = vp/df.shape[0]
    ent = 0
    for val in pp:
        ent = ent + val * math.log(val, nc)
    return -1.0 * ent


def inf_gain(data_frame, nc,child, max_ig_att=None, val=None):
    nr = data_frame.shape[0] #total number of rows in the dataset
    Att_name = data_frame.columns[:-1] #takes in all attribute except the target class
    overall_ent = entropy(data_frame, nc)
    if overall_ent == 0:
        if not max_ig_att==None:
            ET.SubElement(child, "node", entropy=str(-1 * overall_ent), value=val, feature=max_ig_att).text = data_frame['true_value'].unique()[0] #XML command to write the leaf nodes
            return
    else:
        if not max_ig_att==None:
            child = ET.SubElement(child, "node", entropy=str(overall_ent), value=val, feature=max_ig_att) #XML command to write nodes other than leaf node

    ig_list = {}

    for val in Att_name:
        att_val = data_frame[val].unique()
        ent_list = []
        for a_val in att_val: # entropy of child nodes
            tmp_df = data_frame[data_frame[val] == a_val]
            a_size = tmp_df.shape[0]
            ent_sv = entropy(tmp_df, nc)
            ent_sv = (a_size / nr) * ent_sv
            ent_list.append(ent_sv)
        ig = overall_ent - sum(ent_list)
        ig_list[val] = ig

    max_ig_att = None
    max_ig_value = 0

    for key,value in ig_list.items(): # checking the attribute with maximum information gain
        if value > max_ig_value:
            max_ig_value = value
            max_ig_att = key

    uniq_val = data_frame[max_ig_att].unique()
    for val in uniq_val: #distributing the data frame w.r.t attributes passed from each node
        new_df = data_frame[data_frame[max_ig_att] == val]
        new_df = new_df.drop(max_ig_att, axis=1)
        inf_gain(new_df, nc, child, max_ig_att, val)


if __name__ == '__main__': #python statement definition to run the program
    arguments_passed = sys.argv
    properties_dict = {}
    for ind in range(len(arguments_passed)):
        if ind == 0:
            continue
        if '--' in arguments_passed[ind]:
            key = arguments_passed[ind].split('--')[1]
            val = arguments_passed[ind + 1]
            properties_dict[key] = val


data_frame = pd.read_csv(properties_dict['data'], header=None)
data_frame.columns = ['att'+ str(i) for i in data_frame.columns]  # changing the name of columns with 'att'+column number
data_frame = data_frame.rename({data_frame.columns[-1]: 'true_value'}, axis=1)  # changing the name of last column as "true_value"
nc = len(data_frame['true_value'].unique()) # number of classes in target attribute
overall_ent = entropy(data_frame, nc) # parent entropy calculation
root = ET.Element("tree", entropy= str(overall_ent)) # building XML root node

inf_gain(data_frame, nc,root) # information gain for all attributes
tree = ET.ElementTree(root)
tree.write(properties_dict['output']) # writing XML file