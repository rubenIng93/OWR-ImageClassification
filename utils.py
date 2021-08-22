import torch
import csv
import statistics
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle

def plotConfusionMatrix(method, confusionMatrixData):
    fig,ax=plt.subplots(figsize=(10,10))
    sns.heatmap(confusionMatrixData,cmap='seismic',ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix {}".format(method))

    filename = "cm_{}.jpg".format(method) # ex. cm_lwf_30
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, 2)

def map_label_2(map_f, labels):

    mapped_labels = list(map(lambda lb: map_f[lb], labels.cpu().numpy()))
    # back in tensor and cuda
    return torch.LongTensor(mapped_labels).cuda()



def map_label(labels, actual_classes, split=None):
    '''
    Function that maps the label in 
    the range [0, actual_classes-1]:
    '''
    if split != None:
        map = {k: (v + 10 * split) for v, k in enumerate(actual_classes)}
        labels = labels.cpu().numpy()
        for i in range(len(labels)):
            labels[i] = map[labels[i]] 
        return torch.LongTensor(labels).cuda()  # remove cuda if GPU busy
    else:
        map = {k: v for v, k in enumerate(actual_classes)}
        labels = labels.cpu().numpy()
        for i in range(len(labels)):
            labels[i] = map[labels[i]]
        return torch.LongTensor(labels).cuda()  # remove cuda if GPU busy


class FileWriter():
    '''
    Class able to save in a file the result of the training/ testing 
    process
    '''

    def __init__(self, filename, open_world=False):

        self.datafile = open(filename, 'w')

        if not open_world:
            print('seed\tsplit_0\tsplit_1\tsplit_2\tsplit_3\tsplit_4\tsplit_5\tsplit_6\tsplit_7\tsplit_8\tsplit_9',
              file=self.datafile)
        else:
            print('seed\tsplit_0\tsplit_1\tsplit_2\tsplit_3\tsplit_4', file=self.datafile)

    def register_seed(self, accuracy_list):
        print(*accuracy_list, sep='\t', file=self.datafile)

    def close_file(self):
        self.datafile.close()


def trend_chart(filename, title):

    split_0, split_1, split_2, split_3, split_4 = [], [], [], [], []
    split_5, split_6, split_7, split_8, split_9 = [], [], [], [], []

    with open(filename, 'r') as _file:
        values = csv.reader(_file, delimiter='\t')
        header = True
        for row in values:
            if header:
                header = False
            else:
                split_0.append(float(row[1])*100)
                split_1.append(float(row[2])*100)
                split_2.append(float(row[3])*100)
                split_3.append(float(row[4])*100)
                split_4.append(float(row[5])*100)
                split_5.append(float(row[6])*100)
                split_6.append(float(row[7])*100)
                split_7.append(float(row[8])*100)
                split_8.append(float(row[9])*100)
                split_9.append(float(row[10])*100)

    mean_values = []
    list_of_lists = [split_0, split_1, split_2, split_3,
                     split_4, split_5, split_6, split_7, split_8, split_9]

    for _list in list_of_lists:
        mean_values.append(statistics.mean(_list))

    x = np.arange(10)

    fig = go.Figure(data=go.Scatter(x=x, y=mean_values))
    # Edit the layout
    fig.update_layout(title=title,
                      xaxis_title='Split',
                      yaxis_title='Accuracy [%]',
                      xaxis=dict(
                          tickmode='array',
                          tickvals=x                          
                      ))
    fig.show()


def compute_h_mean(closed_dict, open_dict):

    '''
    Compute the harmonic mean
    the input dictionaries have as key the threshold
    (retrieve it as str()!) and as value the accuracy
    
    Returns: a dictionary k=threshold, value=harmonic mean 
    '''

    result = {}

    for t in closed_dict.keys():
        result[str(t)] = 2 / (1/closed_dict[str(t)] + 1/open_dict[str(t)])
    return result



def compute_a_mean(closed_dict, open_dict):
    '''
    Compute the aritmetic mean
    the input dictionaries have as key the threshold
    (retrieve it as str()!) and as value the accuracy
    
    Returns: a dictionary k=threshold, value=harmonic mean 
    '''

    result = {}

    for t in closed_dict.keys():
        result[str(t)] = (closed_dict[str(t)] + open_dict[str(t)]) /2
    return result