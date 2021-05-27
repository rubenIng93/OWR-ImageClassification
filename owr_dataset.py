from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import numpy as np

class Cifar100Dataset(Dataset):

    '''
    The constructor accepts as argument:
    - the split -> (train, test)
    - the seed
    '''
    def __init__(self, split, seed):

        self.split = split
        self.seed = seed
        # dictionary which will contains the random splits 
        # of 10 classes (k=split, v=[random_classes])
        self.subClasses = {}
        self.actual_classes = [] 
        
        # if the split is train perform data augmentation
        # otherwise don't

        if split == 'train':
            self.dataset = torchvision.datasets.CIFAR100(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.RandomCrop(size = 32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
                    # normalized wrt the real cifar100 dataset
                )
            )
        
        else:
            self.dataset = torchvision.datasets.CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
                    # normalized wrt the real cifar100 dataset
                )
            )

    
    def define_splits(self):

        '''
        method that randomly shuffles the classes and 
        produces the splits of 10 classes required for each iteration
        '''

        classes = np.array(range(0, 100)) # 100 = tot n. of classes
        random.seed(self.seed)
        random.shuffle(classes)

        # split the classes in batches of 10 classes each
        # the classes are already shuffled
        for i in range(10):
            self.subClasses[i] = classes[i*10:i*10+10]
        
        # set the first split 
        self.actual_classes = self.subClasses[0]

    
    def concatenate_split(self, i):
        '''
        only for the testset it concatenates classes 10 by 10
        It takes as input the current iteration -> i
        '''
        self.actual_classes = np.concatenate(
            (self.subClasses[i+1], self.actual_classes)
        )

    
    
    def get_imgs_by_target(self):
        '''
        Function that retrieves the indexes of all the images
        belonging to the classes in 'list_of_target'.
        It returns a list of indexes
        '''
        idx = []
        for img_idx, targ in enumerate(self.dataset.targets):
            if targ in self.actual_classes:
                idx.append(img_idx)
        return idx

    def __getitem__(self, index):
        '''
        Args:
        index (int): Index
        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        '''

        image = self.dataset.data[index]
        label = self.dataset.targets[index]

        # the image is already transformed by the CIFAR100 constructor

        return image, label

    def __len__(self):
        '''
        returns the length of the dataset
        '''
        return len(self.dataset)
        

    


