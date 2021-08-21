from tqdm import tqdm
from OWR_Tools.utils import *
from OWR_Tools.resnet import resnet32 as rn32
import numpy as np
from torch.backends import cudnn
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, ConcatDataset
import torch.optim as optim
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy.ma as ma
try:
    import cPickle as pickle
except:
    import pickle


class Loss_Experiments():

    def __init__(self, seeds, file_writer, trainset, testset,
                 epochs, net, splits, b_size, cl_loss, dist_loss):
        '''
        Args:
        - seeds: the list of seeds, aimed to the external loop;
        - file_writer: the FileWriter class to collect the results;
        - trainset: the Training images preprocessed;
        - testset: the Testing images preprocessed;
        - epochs: the number of epochs
        - net: the model, in this case resnet32
        - splits: the number of splits in which the classes are divided
        - scheduler: the training scheduler
        - b_size: the size of batches
        - dist_loss: the distillation loss type: 'l1' or 'l2' or 'lwf'
        - cl_loss: the type of classification loss: 'ce' or 'bce'
        '''

        self.seeds = seeds
        self.writer = file_writer
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.net = net
        self.splits = splits
        self.batch_size = b_size
        self.map = {}
        self.exemplars_set = {}        
        # in case of lwf dist loss, the classification loss must be 
        # BCE
        if dist_loss == 'lwf':
            assert(cl_loss == 'bce')
        self.cl_loss = cl_loss
        self.dist_loss = dist_loss
        self.K = 2000

        # parameters generated by internal funcitons
        self.running_loss_history = []
        self.running_corrects_history = []
        self.accuracy_per_split = []
        self.criterion = ""
        self.train_dataloader = ""
        self.test_dataloader = ""
        self.optimizer = ""
        self.scheduler = ""
        self.all_targets = torch.tensor([])
        self.all_predictions = torch.tensor([])
        self.old_net = ""
        self.current_ex_means = []

        # Optimization of cuda resources
        cudnn.benchmark

    def train(self, split):

        self.map = self.trainset.map

        for e in range(self.epochs):

            # initialize the epoch's metrics
            running_loss = 0.0
            running_corrects = 0.0

            # iterate over the batches
            for inputs, labels in self.train_dataloader:

                # move to GPUs
                inputs = inputs.cuda()
                # print(labels)
                labels = map_label_2(self.map, labels)
                # map the label in range [split * 10, split + 10 * 10]
                # labels = map_label(labels, self.trainset.actual_classes, split)
                # transform it in one hot encoding to fit the BCELoss
                # dimension [batchsize, classes]

                onehot_labels = torch.eye(split*10+10)[labels].to("cuda")

                l1_loss = 0
                l2_loss = 0
                if split > 0:
                    if self.dist_loss == 'l1':
                        l1_loss = self.l1(inputs)
                    elif self.dist_loss == 'l2':
                        l2_loss = self.l2(inputs)
                    elif self.dist_loss == 'lwf':
                        onehot_labels = self.distillation(
                        inputs, onehot_labels, split).cuda()
                   

                # set the network to train mode
                self.net.train()

                # get the score
                outputs = self.net(inputs)
                # compute the loss
                if self.cl_loss == 'ce':
                    loss = self.criterion(outputs, onehot_labels) + l1_loss +l2_loss
                else:
                    loss = self.criterion(outputs, labels) + l1_loss +l2_loss

                # reset the gradients
                self.optimizer.zero_grad()

                # propagate the derivatives
                loss.backward()

                self.optimizer.step()
                # get the predictions
                
                _, preds = torch.max(outputs, 1)
                '''
                preds = self.classify(self.net, inputs)
                '''
                # sum to the metrics the actual scores
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            # compute the epoch's accuracy and loss
            epoch_loss = running_loss/len(self.train_dataloader.dataset)
            epoch_acc = running_corrects.float()/len(self.train_dataloader.dataset)
            self.running_loss_history.append(epoch_loss)
            self.running_corrects_history.append(epoch_acc)

            # display every 5 epochs
            if True:
                print('epoch: {}/{}, LR={}'
                      .format(e+1, self.epochs, self.scheduler.get_last_lr()))
                print('training loss: {:.4f},  training accuracy {:.4f} %'
                      .format(epoch_loss, epoch_acc*100))

            # let the scheduler goes to the next epoch
            self.scheduler.step()

    def classify(self, net, inputs):

        means = {} # the keys are the mapped labels
        # nearest means class classifier
        for label in self.exemplars_set.keys(): 
            loader = DataLoader(self.exemplars_set[label], batch_size=len(self.exemplars_set[label])
                                )
            with torch.no_grad():
                for img, _ in loader: # a single batch
                    img = img.cuda()
                    net = net.cuda()
                    features = net.extract_features(img)
                    features = features / features.norm()
                    mean = torch.mean(features, 0) # this is the mean of all images in the same class exemplars
                    means[label] = mean

        # assing the class to the inputs
        norms = []
        features = net.extract_features(inputs)
        for k in means.keys(): 
            mean_k = means[k]
            mean_k = mean_k/mean_k.norm()            
            norm = torch.norm((features - mean_k), dim=1)
            #print(f"Norm shape: {norm.shape}")
            norms.append(norm)
        
        norms = torch.stack(norms)
        preds = torch.argmin(norms, dim=0)

        return preds.cuda()

    def run_loop(self):

        for seed in self.seeds:
            # define the splits according to the seed
            self.trainset.define_splits(seed)
            self.testset.define_splits(seed)
            # setting the seed
            np.random.seed(seed)
            torch.manual_seed(seed)

            # initialize the accuracies array
            self.accuracy_per_split.append(seed)
            # reset the net
            self.net = rn32().cuda()
            if self.cl_loss == 'bce':
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

            # the 10 iterations for finetuning, 10 classes each
            for split in range(0, self.splits):

                # defining the proper set of classes for training
                self.trainset.change_subclasses(split)  # update the subclasses
                train_subset = Subset(
                    self.trainset, self.trainset.get_imgs_by_target())
                # update representation adding the exemplars
                temp = []
                if bool(self.exemplars_set):
                    # if there is something in the exemplar set
                    for l in self.exemplars_set.values():
                        temp.extend(l)

                # extend the dataset with the exemplars
                updated_train_subset = ConcatDataset((train_subset, temp))
                # prepare the dataloader
                self.train_dataloader = DataLoader(updated_train_subset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True, num_workers=2)

                # testset preparation
                if split == 0:
                    self.testset.change_subclasses(split)
                else:
                    # concatenate split
                    self.testset.concatenate_split(split)

                test_subset = Subset(
                    self.testset, self.testset.get_imgs_by_target())

                self.test_dataloader = DataLoader(test_subset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)

                # start the training procedure
                print(
                    5*"*"+f" Training the for classes {split*10} : {split*10+10} " + 5*"*"+"\n")

                if split > 0:
                    # save the old trained network in case of lwf or icarl
                    self.old_net = copy.deepcopy(self.net)
                    # move the old net to GPUs
                    self.old_net.cuda()
                    # set up the resnet with the proper number of outputs neurons in
                    # the final fully connected layer
                    out_neurons = split*10+10  # new number of output classes
                    in_features = self.net.fc.in_features  # n. of in features in the fc
                    weight = self.net.fc.weight.data  # current weights in the fc
                    # new fc with proper n. of classes
                    self.net.fc = nn.Linear(in_features, out_neurons)
                    # keep the old weights
                    self.net.fc.weight.data[:split*10] = weight
                    self.net.cuda()
                    # reduce the exemplars set
                    self.reduce_exemplar_set(split)

                parameters_to_optimize = self.net.parameters()
                self.optimizer = optim.Adam(parameters_to_optimize, lr=0.01,
                                            weight_decay=0.00001)
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, [49, 63], gamma=0.05)

                self.running_loss_history = []
                self.running_corrects_history = []

                
                # train
                self.train(split)
                # update representation
                self.build_exemplars_set(self.trainset, split)
                # test
                self.test(split)

            # register the seed's results
            self.writer.register_seed(self.accuracy_per_split)

        # close the file writer
        self.writer.close_file()

    '''
    Losses
    '''
    
    def l1(self, inputs):
        self.old_net.eval()
        self.net.eval()
        with torch.no_grad():
            features = self.net.extract_features(inputs)
            old_features = self.old_net.extract_features(inputs)

        l1loss = nn.L1Loss()

        l1_loss = l1loss(features, old_features)
        

        return l1_loss

    def l2(self, inputs):
        self.old_net.eval()
        self.net.eval()
        with torch.no_grad():
            features = self.net.extract_features(inputs)
            old_features = self.old_net.extract_features(inputs)

        l2loss = nn.MSELoss()

        l2_loss = l2loss(features, old_features)
        #l2_loss.backward()

        return l2_loss    

    def distillation(self, inputs, new_onehot_labels, split):
        m = nn.Sigmoid()
        # compute the old network's outputs for the new classes
        old_outputs = self.old_net(inputs)
        # apply them the sigmoid function
        old_outputs = m(old_outputs).cuda()
        # substitute the true labels with the outputs of the
        # previous step for the classes in the previous split
        new_onehot_labels[:, 0:split*10] = old_outputs
        return new_onehot_labels

    def build_exemplars_set(self, trainset, split):

        print(f'Building exemplars split {split}')

       # initialize the data structures
        classes_means = {}
        features = {}
        #exemplars = {}

        self.net.eval()
        with torch.no_grad():
            # actual classes are the 10 new classes
            for act_class in trainset.actual_classes:

                # get all the images belonging to the current label
                actual_idx = trainset.get_imgs_by_chosing_target(act_class)
                # build a subset and a dataloader to better manage the images
                subset = Subset(trainset, actual_idx)
                loader = DataLoader(subset, batch_size=len(subset))
                # get the mapped label of the actual class
                mapped_label = trainset.map[act_class]
                

                # extract the features of the images and take the class mean
                for img, _ in loader:
                    img = img.cuda()
                    img = self.net.extract_features(img)
                    img = img / img.norm()
                    features[mapped_label] = img.cpu().numpy()
                    mean = torch.mean(img, 0)  # mean by column
                    classes_means[mapped_label] = mean.cpu().numpy()

                exemplar = []
                cl_mean = np.zeros((1, 64))
                so_far_classes = split * 10 + 10
                m = int(self.K / so_far_classes)
                # apply the paper algorithm
                indexes = []
                i = 0
                for i in range(m):
                    if i > 0:
                        cl_mean += features[mapped_label][index]
                        # take the best as image, not features
                    x = classes_means[mapped_label] - (cl_mean + features[mapped_label]) / (i+1)
                    # print(x.shape)
                    x = np.linalg.norm(x, axis=1)
                    # masking for avoiding duplicated
                    mask = np.zeros(len(x), int)
                    mask[indexes] = 1
                    x_masked = ma.masked_array(x, mask=mask)
                    # print(x.shape)
                    index = np.argmin(x_masked)                    
                    indexes.append(index)                        
                    exemplar.append(loader.dataset[index])

                #print(np.unique(indexes, return_counts=True))
                
                self.exemplars_set[mapped_label] = exemplar

            #self.exemplars_set = exemplars

    def reduce_exemplar_set(self, split):
        '''
        Called starting from the 2nd split after having 
        computed the outputs and before updating the exemplars set
        '''
        # m is the new target cardinality for each exemplar set
        current_m = self.K / (split * 10)
        new_m = self.K / (split * 10 + 10)

        to_remove = int(current_m - new_m)

        for k in self.exemplars_set.keys():
            self.exemplars_set[k] = self.exemplars_set[k][:-to_remove]

    def test(self, split):

        # save prediction and targets to get the conf matrix
        all_targets = torch.tensor([])
        self.all_targets = all_targets.type(torch.LongTensor)
        all_predictions = torch.tensor([])
        self.all_predictions = all_predictions.type(torch.LongTensor)

        # set the network to test mode
        self.net.train(False)
        # initialize the metric for test
        running_corrects_test = 0

        # iterate over the test dataloader
        for images, targets in tqdm(self.test_dataloader):
            # move to GPUs
            images = images.cuda()
            targets = targets.cuda()
            # map the label in range [0, n_classes - 1]
            # print(targets)
            targets = map_label_2(self.map, targets)
            # print(targets)
            # get the predictions

            preds = self.classify(self.net, images)
            '''
            outputs = self.net(images)
            # get the predictions
            _, preds = torch.max(outputs, 1)            
            '''
            self.all_targets = torch.cat(
                (self.all_targets.cuda(), targets.cuda()), dim=0)
            self.all_predictions = torch.cat(
                (self.all_predictions.cuda(), preds.cuda()), dim=0)
            # sum the actual scores to the metric
            running_corrects_test += torch.sum(preds == targets)

        # calculate the accuracy
        accuracy = running_corrects_test / \
            float(len(self.test_dataloader.dataset))
        # update the global metric
        self.accuracy_per_split.append(accuracy.cpu().numpy())
        # display the accuracy
        print(f'Test Accuracy for classes {0} to {split*10+10}: {accuracy}\n')

        confusionMatrixData = confusion_matrix(
            self.all_targets.cpu().numpy(),
            self.all_predictions.cpu().numpy()
        )
        plotConfusionMatrix(f"{self.cl_loss.upper()} + {self.dist_loss.upper()}", confusionMatrixData)
        if split == 9:
            with open(f'cm_data_{self.cl_loss}_{self.dist_loss}.pth', 'wb') as f:
                pickle.dump(confusionMatrixData, f, 2)
