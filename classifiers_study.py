from tqdm import tqdm
from OWR_Tools.utils import *
from OWR_Tools.cosine_resnet import resnet32 as cos_rn32, CosineLinear, SplitCosineLinear
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
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import math
import torch.nn.functional as F
import numpy.ma as ma
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import cPickle as pickle
except:
    import pickle


'''
functions for combo, hooks
'''
cur_features = []
ref_features = []
old_scores = []
new_scores = []

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs



class CSEnvironment():

    def __init__(self, seeds, file_writer, trainset, testset,
                 epochs, net, splits, b_size, classifier):
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
        - mode: 'finetuning', 'lwf' or 'icarl'
        - classifier: choose FC, NME, KNN or combo
            the last one follows the implementation details as:
                @InProceedings{Hou_2019_CVPR,
                author = {Hou, Saihui and Pan, Xinyu and Loy, Chen Change and Wang, Zilei and Lin, Dahua},
                title = {Learning a Unified Classifier Incrementally via Rebalancing},
                booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                month = {June},
                year = {2019}
                }

        '''

        self.seeds = seeds
        self.writer = file_writer
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.net = net
        self.splits = splits
        self.batch_size = b_size
        self.classifier = classifier
        self.map = {}
        self.exemplars_set = {}
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

    '''
    Classifiers
    '''

    def NME_classify(self, net, inputs):

        means = {}  # the keys are the mapped labels
        # nearest means class classifier
        for label in self.exemplars_set.keys():
            loader = DataLoader(self.exemplars_set[label], batch_size=len(self.exemplars_set[label])
                                )
            with torch.no_grad():
                for img, _ in loader:  # a single batch
                    img = img.cuda()
                    net = net.cuda()
                    features = net.extract_features(img)
                    features = features / features.norm()
                    # this is the mean of all images in the same class exemplars
                    mean = torch.mean(features, 0)
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

    def KNN_classify(self, net, inputs):

        with torch.no_grad():
            features = net.extract_features(inputs)
            features = features / features.norm()
            preds = self.knn.predict(features.cpu().numpy())
            # back in tensor
            preds = torch.Tensor(preds).cuda()

        return preds

    '''
    MAIN LOOP
    '''

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
            if self.classifier == 'combo':
                self.net = cos_rn32().cuda()
                self.criterion = nn.CrossEntropyLoss()
            else:            
                self.net = rn32().cuda()
                self.criterion = nn.BCEWithLogitsLoss()

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

                cur_lamb = 5

                parameters_to_optimize = self.net.parameters()

                if split > 0:
                    # save the old trained network in case of lwf or icarl
                    self.old_net = copy.deepcopy(self.net)
                    # freeze the network
                    #for p in self.old_net.parameters():
                        #p.requires_grad = False
                    # move the old net to GPUs
                    self.old_net.cuda()
                    # set up the resnet with the proper number of outputs neurons in
                    # the final fully connected layer
                    out_neurons = split*10+10  # new number of output classes

                    if self.classifier != 'combo':
                        in_features = self.net.fc.in_features  # n. of in features in the fc
                        weight = self.net.fc.weight.data  # current weights in the fc
                        # new fc with proper n. of classes
                        self.net.fc = nn.Linear(in_features, out_neurons)
                        # keep the old weights
                        self.net.fc.weight.data[:split*10] = weight
                        self.net.cuda()

                    elif self.classifier == 'combo' and split == 1:
                        in_features = self.net.cosine.in_features  # n. of in features in the fc
                        out_features = self.net.cosine.out_features # n. out features
                        weight = self.net.cosine.weight.data  # current weights in the fc
                        sigma = self.net.cosine.sigma.data
                        # new fc with proper n. of classes
                        self.net.cosine = SplitCosineLinear(
                            in_features, out_features, 10)
                        # keep the old weights
                        self.net.cosine.fc1.weight.data = weight
                        self.net.cosine.sigma.data = sigma
                        self.lambd = out_features*1.0 / 10
                        ignored_params = list(map(id, self.net.cosine.fc1.parameters()))
                        base_params = filter(lambda p: id(p) not in ignored_params, \
                            self.net.parameters())
                        parameters_to_optimize = [{'params': base_params, 'lr': 0.01, 'weight_decay': 1e-5}, \
                        {'params': self.net.cosine.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
                        self.net.cuda()

                    elif self.classifier == 'combo' and split > 1:
                        in_features = self.net.cosine.in_features
                        out_features1 = self.net.cosine.fc1.out_features
                        out_features2 = self.net.cosine.fc2.out_features
                        sigma = self.net.cosine.sigma.data

                        new_fc = SplitCosineLinear(
                            in_features, out_features1+out_features2, 10)
                        new_fc.fc1.weight.data[:out_features1] = self.net.cosine.fc1.weight.data
                        new_fc.fc1.weight.data[out_features1:] = self.net.cosine.fc2.weight.data
                        new_fc.sigma.data = self.net.cosine.sigma.data
                        self.net.cosine = new_fc
                        self.lambd = (out_features1 + out_features2)*1.0 / 10
                        ignored_params = list(map(id, self.net.cosine.fc1.parameters()))
                        base_params = filter(lambda p: id(p) not in ignored_params, \
                            self.net.parameters())
                        parameters_to_optimize = [{'params': base_params, 'lr': 0.01, 'weight_decay': 1e-5}, \
                        {'params': self.net.cosine.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
                        self.net.cuda()

                    # reduce the exemplars set
                    self.reduce_exemplar_set(split)

                
                if split > 0 and self.classifier == 'combo':
                    self.lamb = 5 * math.sqrt(self.lambd)
                else:
                    self.lamb = 5

                self.optimizer = optim.Adam(parameters_to_optimize, lr=0.01,
                                            weight_decay=0.00001)
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, [49, 63], gamma=0.05)

                self.running_loss_history = []
                self.running_corrects_history = []

                # train
                self.train(split)
                # train the knn on exemplars if it's the choice
                
                # update representation
                self.build_exemplars_set(self.trainset, split)

                if self.classifier == 'KNN':
                    self.train_KNN(50)
                # test
                self.test(split)

            # register the seed's results
            self.writer.register_seed(self.accuracy_per_split)

        # close the file writer
        self.writer.close_file()

    '''
    TRAIN
    '''

    def train(self, split):

        self.map = self.trainset.map

        if split > 0 and self.classifier == 'combo':
            self.old_net.eval()
            # hooks
            num_old_classes = self.old_net.cosine.out_features
            handle_ref_features = self.old_net.cosine.register_forward_hook(get_ref_features)
            handle_cur_features = self.net.cosine.register_forward_hook(get_cur_features)
            handle_old_scores_bs = self.net.cosine.fc1.register_forward_hook(get_old_scores_before_scale)
            handle_new_scores_bs = self.net.cosine.fc2.register_forward_hook(get_new_scores_before_scale)

        for e in range(self.epochs):

            # initialize the epoch's metrics
            running_loss = 0.0
            running_corrects = 0.0

            # iterate over the batches
            for inputs, labels in self.train_dataloader:

                # move to GPUs
                inputs = inputs.cuda()
                # reset the gradients
                self.optimizer.zero_grad()
                # print(labels)
                labels = map_label_2(self.map, labels)
                # map the label in range [split * 10, split + 10 * 10]
                # labels = map_label(labels, self.trainset.actual_classes, split)
                # transform it in one hot encoding to fit the BCELoss
                # dimension [batchsize, classes]
                onehot_labels = torch.eye(split*10+10)[labels].to("cuda")

                cosineL = torch.zeros(1).cuda()
                marginL = torch.zeros(1).cuda()
                #num_old_classes = 10*split
                K = 2

                # set the network to train mode
                self.net.train()

                # get the score
                outputs = self.net(inputs)

                if split > 0:

                    if self.classifier != 'combo':
                        # use the exemplars coming from the previous step
                        onehot_labels = self.distillation(
                            inputs, onehot_labels, split).cuda()
                    else:
                        with torch.no_grad():

                            self.old_net.eval()

                            old_features, old_outputs = self.old_net.forward_with_features(inputs)
                            old_outputs = old_outputs.cuda()
                            old_features = old_features.detach()

                            cosineL = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
									                          torch.ones(inputs.shape[0]).cuda()) * self.lamb

                            # scores before scale
                            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                            #print(self.net.cosine.fc1.in_features, self.net.cosine.fc1.out_features)
                            #print(self.net.cosine.fc2.in_features, self.net.cosine.fc2.out_features)
                            #print(old_scores.size(), new_scores.size(), outputs_bs.size(), outputs.size())
                            assert(outputs_bs.size()==outputs.size())
                            gt_index = torch.zeros(outputs_bs.size()).cuda()
                            gt_index = gt_index.scatter(1, labels.cuda().view(-1, 1), 1).ge(0.5)
                            gt_scores = outputs_bs.masked_select(gt_index)
                            # get top-K scores on novel classes
                            max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
                            # cosine distillation
                            #cosineL = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                                #torch.ones(inputs.shape[0]).to(device)) * self.lamda
                            #lam = 5 * np.sqrt(10/(num_old_classes))
                            #cosineL = self.cosine(inputs, self.lambd)
                            # margin loss
                            old_idx = labels.lt(num_old_classes)
                            old_num = torch.nonzero(old_idx).size(0)
                            if old_num > 0:
                                gt_scores = gt_scores[old_idx].view(-1, 1).repeat(1, K)
                                max_novel_scores = max_novel_scores[old_idx]
                                assert(gt_scores.size() == max_novel_scores.size())
                                assert(gt_scores.size(0) == old_num)
                                marginL = nn.MarginRankingLoss(margin=0.5)(gt_scores.view(-1, 1),\
                                    max_novel_scores.view(-1, 1), torch.ones(old_num*K).cuda())

                

                # compute the loss
                if self.classifier == 'combo':
                    loss = self.criterion(outputs, labels) + cosineL + marginL
                else:
                    loss = self.criterion(outputs, onehot_labels)

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

        if split > 0 and self.classifier == 'combo':
            handle_ref_features.remove()
            handle_cur_features.remove()
            handle_old_scores_bs.remove()
            handle_new_scores_bs.remove()

    def train_KNN(self, k):
        '''
        k: the number of nearest neighbors
        '''
        print('Train KNN')
        exemplars = []
        for label in self.exemplars_set.keys():
            exemplars.extend(self.exemplars_set[label])

        loader = DataLoader(exemplars, batch_size=self.batch_size)
        self.knn = KNeighborsClassifier(k)
        # requires the mapping
        with torch.no_grad():
            features = []
            labels = []
            for images, lbs in loader:
                images = images.cuda()
                lbs = map_label_2(self.map, lbs)
                ext_features = self.net.extract_features(images)
                ext_features = ext_features/ext_features.norm()
                features.append(ext_features)
                labels.append(lbs)
            torch_features = torch.cat(features)
            torch_labels = torch.cat(labels)
            np_features = torch_features.cpu().numpy()
            np_labels = torch_labels.cpu().numpy()
            self.knn.fit(np_features, np_labels)

            ## Visualization through t-SNE
            X_red = TSNE(n_components=2).fit_transform(np_features)
            fig, ax = plt.subplots(figsize=(15,10))
            #ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_red[:,0], X_red[:,1], c=np_labels, cmap='tab20', s=2)
            #ax.legend(*scatter.legend_elements(),title="Classes")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.title('t-SNE 2D Exemplars features visualization')
            plt.show()


            

    '''
    TEST
    '''

    def test(self, split):

        print(f'Test split {split}')

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

            if self.classifier == 'FC' or self.classifier == 'combo':
                outputs = self.net(images)
                _, preds = torch.max(outputs, 1)

            if self.classifier == 'KNN':
                preds = self.KNN_classify(self.net, images)

            elif self.classifier == 'NME':
                preds = self.NME_classify(self.net, images)

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
        plotConfusionMatrix("Finetuning", confusionMatrixData)
        if split == 9:
            with open(f'cm_data_{self.classifier}.pth', 'wb') as f:
                pickle.dump(confusionMatrixData, f, 2)

    '''
    EXEMPLARS MANAGEMENT
    '''

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

        if (self.K % (split * 10 + 10)) != 0:
            to_remove = int(current_m - new_m) + 1
        else:
            to_remove = int(current_m - new_m)

        for k in self.exemplars_set.keys():
            self.exemplars_set[k] = self.exemplars_set[k][:-to_remove]

    '''
    DISTILLATIONS
    '''

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

    def cosine(self, inputs, lmbd):
        features = self.net.extract_features(inputs)
        old_old_features = self.old_net.forward_with_features(inputs)
        cosineLoss = nn.CosineEmbeddingLoss()(features, old_features,
                                              torch.ones(inputs.shape[0]).cuda())
        lg_dis = cosineLoss * lmbd
        return lg_dis
