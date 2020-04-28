import numpy as np
import re
import csv
import string
import re
import xlrd
from xlutils.copy import copy
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    cacheStopWords = []
    for line in open("word.txt", "r"):  # 设置文件对象并读取每一行文件
        cacheStopWords.append(line)

    # Load data from files

    x_text = []
    y = []
    birth_weight_file = 'challenge_item.xls'
    data = xlrd.open_workbook(birth_weight_file)
    table = data.sheet_by_index(0)
    titleFile = []
    titleFile.append(table.col_values(1))
    requirment = []
    requirment.append(table.col_values(2))
    technology = []
    technology.append(table.col_values(3))
    lan = []
    lan.append(table.col_values(4))
    for i in range(1, len(titleFile[0])):
        labels = [0, 0, 0, 0, 0, 0]
        text1 = ''.join([word + " " for word in titleFile[0][i].split() if word not in cacheStopWords])
        text2 = ''.join([word + " " for word in requirment[0][i].split() if word not in cacheStopWords])
        x_text.append(text1+' '+text2)
        if 'Java' in technology[0][i].split(',') or 'Java' in lan[0][i].split(','):
            labels[0] = 1
        if 'JavaScript' in technology[0][i].split(',') or 'JavaScript' in lan[0][i].split(','):
            labels[1] = 1
        if 'HTML' in technology[0][i].split(',') or 'HTML' in lan[0][i].split(','):
            labels[2] = 1
        if 'CSS' in technology[0][i].split(',') or 'CSS' in lan[0][i].split(','):
            labels[3] = 1
        if 'J2EE' in technology[0][i].split(',') or 'J2EE' in lan[0][i].split(','):
            labels[4] = 1
        if 'Spring' in technology[0][i].split(',') or 'Spring' in lan[0][i].split(','):
            labels[5] = 1
        y.append(labels)
    x_text = [s.strip() for s in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    y = np.array(y)
    return [x_text, y]

def load_data_and_labels2():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    cacheStopWords = []
    for line in open("word.txt", "r"):  # 设置文件对象并读取每一行文件
        cacheStopWords.append(line)

    #y = np.concatenate([positive_labels, negative_labels], 0)
    x_text = []
    y = []
    birth_weight_file = 'eval.xls'
    data = xlrd.open_workbook(birth_weight_file)
    table = data.sheet_by_index(0)
    titleFile = []
    titleFile.append(table.col_values(1))
    requirment = []
    requirment.append(table.col_values(2))
    technology = []
    technology.append(table.col_values(3))
    lan = []
    lan.append(table.col_values(4))
    for i in range(1, len(titleFile[0])):
        text1 = ''.join([word + " " for word in titleFile[0][i].split() if word not in cacheStopWords])
        text2 = ''.join([word + " " for word in requirment[0][i].split() if word not in cacheStopWords])
        x_text.append(text1 + ' ' + text2)
    x_text = [s.strip() for s in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    y = np.array(y)
    return [x_text, y]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def cal_metric(predicted_labels, labels):
    """
    Calculate the metric(recall, accuracy, F, etc.).
    Args:
        predicted_labels: The predicted_labels
        labels: The true labels
    Returns:
        The value of metric
    """
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    l1 = len(label_no_zero)
    l2 = len(predicted_labels)
    if l1 == 0:
        rec = 0.9
    else:
        rec = count / l1
    if l2 == 0:
        acc = 0.9
    else:
        acc = count / l2
    if (rec + acc) == 0:
        F = 0.0
    else:
        F = (2 * rec * acc) / (rec + acc)
    return rec, acc, F

def get_label_using_scores_by_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict value greater than threshold, then choose the label which has the max predict value.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_values: The predicted values
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        value_list = []
        for index, predict_value in enumerate(score):
            if predict_value > threshold:
                index_list.append(index)
                value_list.append(predict_value)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            value_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_values.append(value_list)
    return predicted_labels, predicted_values

def get_label_using_scores_by_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        value_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            value_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_values.append(value_list)
    return predicted_labels, predicted_values
