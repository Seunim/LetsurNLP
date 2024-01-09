import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict

FILE_DIR = 'data/text_total.json'
total_len = []
np.random.seed(7)

def split_train_dev_test():
    f = open('data/text_total.json', 'r')
    data = f.readlines()
    f.close()
    id = [i for i in range(103882)]
    np_data = np.array(data)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.05, random_state=0)
    train, val = train_test_split(train, test_size=0.05, random_state=0)
    train = list(train)
    val = list(val)
    test = list(test)
    f = open('data/text_train.json', 'w')
    f.writelines(train)
    f.close()
    f = open('data/text_test.json', 'w')
    f.writelines(test)
    f.close()
    f = open('data/text_val.json', 'w')
    f.writelines(val)
    f.close()

    print(len(train), len(val), len(test))
    return

def remove_empty(d):
    if isinstance(d, dict):
        return {k: remove_empty(v) for k, v in d.items() if v}
    else:
        return d


def get_hierarchy():
    f = open('data/text_total.json', 'r')
    data = f.readlines()
    f.close()

    form = lambda x: {"doc_label": x}

    doc_label = []

    for line in data:
        line = line.rstrip('\n')
        line = json.loads(line)
        
        line = line['doc_label']
        #print(line)
        doc_label.append(line)
    
    data = map(form, doc_label)


    label_hierarchy = defaultdict(list)
    for i in data:
        labels = ['Root'] + i["doc_label"]
        for j, k in zip(labels, labels[1:]):
            if k not in label_hierarchy[j]:
                label_hierarchy[j].append(k)
                
    label_hierarchy = remove_empty(label_hierarchy)
    print(label_hierarchy)
    
    f = open('data/text.taxonomy', 'w')

    for i in label_hierarchy.keys():
        line = [i]
        line.extend(label_hierarchy[i])
        line = '\t'.join(line) + '\n'
        f.write(line)
    f.close()

    
if __name__ == '__main__':
    get_hierarchy()
    split_train_dev_test()