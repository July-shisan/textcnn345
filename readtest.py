import csv
import os
import numpy as np
import random
# name of data file
# 数据集名称
birth_weight_file = '/home/tangsu/PycharmProjects/TEXT/labeledTrainData.tsv'
#csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)
with open(birth_weight_file,) as csvfile:
    file_list = csv.reader(csvfile,'mydialect')
    for line in file_list:
        print(line[1])
        if line[1]=='0':
            print('1#')

csv.unregister_dialect('mydialect')