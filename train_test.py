from tqdm import tqdm
from utils import map_label

DEVICE = 'cuda'

# Optimization of cuda resources
cudnn.benchmark

class TrainTester():

    def __init__(self, seeds, file_writer, trainset, testset,
    epochs, net, splits, scheduler ):

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
        '''

        self.seeds = seeds
        self.writer = file_writer
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.net = net
        self.splits = splits
        self.scheduler = scheduler

        # parameters generated by internal funcitons
        self.running_loss_history = []
        self.running_corrects_history = []
        self.accuracy_per_split = []
        self.criterion
        self.train_dataloader
        self.test_dataloader
        self.optimizer
        


    def train(self, split):

        for e in range(self.epochs):

            # initialize the epoch's metrics
            running_loss = 0.0
            running_corrects = 0.0

            # iterate over the batches
            for input, labels in self.train_dataloader:

                # move to GPUs
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE) 
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
            if e % 5 == 0:
                print('epoch: {}/{}, LR={}'
                    .format(e+1, self.epochs, self.scheduler.get_last_lr()))
                print('training loss: {:.4f},  training accuracy {:.4f} %'
                    .format(epoch_loss, epoch_acc*100))

            # let the scheduler goes to the next epoch
            self.scheduler.step()


    def test(self, split):

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
            # sum the actual scores to the metric
            running_corrects_test += torch.sum(preds == targets)

        # calculate the accuracy
        accuracy = running_corrects_test / float(len(self.test_dataloader.dataset))
        # update the global metric
        self.accuracy_per_split.append(accuracy.cpu().numpy())
        # display the accuracy
        print(f'Test Accuracy for classes {0} to {split*10+10}: {accuracy}\n')




        
