import torch


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
         file=filename)

    def register_seed(self, accuracy_list):
        print(*accuracy_list, sep='\t', file=self.datafile)

    def close_file(self):
        self.datafile.close()