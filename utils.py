import torch
import csv
import statistics
import plotly.graph_objects as go
import numpy as np


def map_label(labels, actual_classes):
    '''
    Function that maps the label in 
    the range [0, actual_classes-1]:
    '''
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

    def __init__(self, filename):
        
        self.datafile = open(filename, 'w')

        print('seed\tsplit_0\tsplit_1\tsplit_2\tsplit_3\tsplit_4\tsplit_5\tsplit_6\tsplit_7\tsplit_8\tsplit_9',
         file=self.datafile)

    def register_seed(self, accuracy_list):
        print(*accuracy_list, sep='\t', file=self.datafile)

    def close_file(self):
        self.datafile.close()

def trend_chart(filename, title):

    split_0, split_1, split_2, split_3, split_4 = [], [], [], [], []
    split_5, split_6, split_7, split_8, split_9 = [], [], [], [], []    

    with open(filename, 'r') as _file:
        values = csv.reader(_file, delimiter='/t')
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
                   yaxis_title='Accuracy [%]')
    fig.show()

    

    

