# class means for the actual classes
exemplars_means = {}
features = {}

for act_class in trainset.actual_classes:

  actual_idx = trainset.get_imgs_by_chosing_target(act_class)
  subset = Subset(trainset, actual_idx)
  loader = DataLoader(subset, batch_size=len(subset))

  for img, _ in loader:
    img = img.cuda()
    img = resnet32.extract_features(img)
    features[act_class] = img.detach().cpu().numpy()
    mean = torch.mean(img, 0) # column
    exemplars_means[act_class] = mean.detach().cpu().numpy()


# build exemplars set for a single class
exemplar = []
cl_mean = np.zeros((1, 64))

for i in range(3): # must be m
  x = exemplars_means[4] - (cl_mean + features [4]) / i+1
  print(x.shape)
  x = np.linalg.norm(x, axis=1)
  print(x.shape)
  index = np.argmin(x)
  print(index)
  cl_mean += features[4][index]
  exemplar.append(loader.dataset[index])
