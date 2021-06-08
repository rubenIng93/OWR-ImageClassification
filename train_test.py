from tqdm import tqdm
from OWR_Tools.utils import map_label
from OWR_Tools.resnet import resnet32 as rn32
import numpy as np
from torch.backends import cudnn
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim


class TrainTester():

    def __init__(self, seeds, file_writer, trainset, testset,
                 epochs, net, splits, b_size):
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
        '''

        self.seeds = seeds
        self.writer = file_writer
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.net = net
        self.splits = splits
        self.batch_size = b_size

        # parameters generated by internal funcitons
        self.running_loss_history = []
        self.running_corrects_history = []
        self.accuracy_per_split = []
        self.criterion = ""
        self.train_dataloader = ""
        self.test_dataloader = ""
        self.optimizer = ""
        self.scheduler = ""
        self.all_targets = ""
        self.all_predictions = ""

        # Optimization of cuda resources
        cudnn.benchmark

    def train(self, split):

        for e in range(self.epochs):

            # initialize the epoch's metrics
            running_loss = 0.0
            running_corrects = 0.0

            # iterate over the batches
            for inputs, labels in self.train_dataloader:

                # move to GPUs
                inputs = inputs.cuda()
                labels = labels.cuda()
                # map the label in range [0, n_classes - 1]
                labels = map_label(labels, self.trainset.actual_classes)
                # transform it in one hot encoding to fit the BCELoss
                onehot_labels = torch.eye(split*10+10)[labels].to("cuda")

                # set the network to train mode
                self.net.train()
                # get the score
                outputs = self.net(inputs)
                # compute the loss
                loss = self.criterion(outputs, onehot_labels)
                # reset the gradients
                self.optimizer.zero_grad()
                # propagate the derivatives
                loss.backward()

                self.optimizer.step()
                # get the predictions
                _, preds = torch.max(outputs, 1)
                # sum to the metrics the actual scores
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            # compute the epoch's accuracy and loss
            epoch_loss = running_loss/len(self.train_dataloader.dataset)
            epoch_acc = running_corrects.float()/len(self.train_dataloader.dataset)
            self.running_loss_history.append(epoch_loss)
            self.running_corrects_history.append(epoch_acc)

            # display every 5 epochs
            if (e+1) % 5 == 0:
                print('epoch: {}/{}, LR={}'
                      .format(e+1, self.epochs, self.scheduler.get_last_lr()))
                print('training loss: {:.4f},  training accuracy {:.4f} %'
                      .format(epoch_loss, epoch_acc*100))

            # let the scheduler goes to the next epoch
            self.scheduler.step()

    def test(self, split):

        # save prediction and targets to get the conf matrix
        all_targets = []
        all_predictions = []

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
            targets = map_label(targets, self.testset.actual_classes)
            # forward pass
            outputs = self.net(images)
            # get the predictions
            _, preds = torch.max(outputs, 1)
            # concatenate to the global lists
            all_predictions.extend(preds.tolist())
            all_targets.extend(targets.data.tolist())
            #self.all_targets = np.concatenate(all_targets, targets.cpu().numpy())
            #self.all_predictions = np.concatenate(all_predictions, preds.cpu().numpy())
            # sum the actual scores to the metric
            running_corrects_test += torch.sum(preds == targets)

        # calculate the accuracy
        accuracy = running_corrects_test / \
            float(len(self.test_dataloader.dataset))
        # update the global metric
        self.accuracy_per_split.append(accuracy.cpu().numpy())
        # display the accuracy
        print(f'Test Accuracy for classes {0} to {split*10+10}: {accuracy}\n')

    def run_loop(self):

        for seed in self.seeds:
            # define the splits according to the seed
            self.trainset.define_splits(seed)
            self.testset.define_splits(seed)

            # initialize the accuracies array
            self.accuracy_per_split.append(seed)
            # reset the net
            self.net = rn32().cuda()
            self.criterion = nn.BCEWithLogitsLoss()

            # the 10 iterations for finetuning, 10 classes each
            for split in range(0, self.splits):

                # defining the proper set of classes for training
                self.trainset.change_subclasses(split)  # update the subclasses
                train_subset = Subset(
                    self.trainset, self.trainset.get_imgs_by_target())
                # defining the proper set of classes for testing
                if split == 0:
                    self.testset.change_subclasses(split)
                else:
                    # concatenate at each iteration the old split
                    for i in range(0, split):
                        self.testset.concatenate_split(i)

                test_subset = Subset(
                    self.testset, self.testset.get_imgs_by_target())

                # prepare the dataloaders
                self.train_dataloader = DataLoader(train_subset, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=2)
                self.test_dataloader = DataLoader(test_subset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)

                # start the training procedure
                print(
                    5*"*"+f" Training the for classes {split*10} : {split*10+10} " + 5*"*"+"\n")

                parameters_to_optimize = self.net.parameters()
                self.optimizer = optim.SGD(parameters_to_optimize, lr=2,
                                    momentum=0.9, weight_decay=0.00001)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [49, 63], gamma=0.2)

                if split > 0:
                    # set up the resnet with the proper number of outputs neurons in
                    # the final fully connected layer
                    out_neurons = split*10+10  # new number of output classes
                    in_features = self.net.fc.in_features  # n. of in features in the fc
                    weight = self.net.fc.weight.data  # current weights in the fc
                    # new fc with proper n. of classes
                    self.net.fc = nn.Linear(in_features, out_neurons)
                    self.net.fc.weight.data[:split*10] = weight  # keep the old weights
                    self.net.cuda()

                self.running_loss_history = []
                self.running_corrects_history = []
                
                # train 
                self.train(split)
                # test
                self.test(split)
            
            # register the seed's results
            self.writer.register_seed(self.accuracy_per_split)
        
        # close the file writer
        self.writer.close_file()
        
