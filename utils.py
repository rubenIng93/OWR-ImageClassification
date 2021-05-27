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
    return torch.LongTensor(labels).cuda() # remove cuda if GPU busy
