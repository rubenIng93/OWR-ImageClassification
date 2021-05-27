# OWR-ImageClassification
Open World Recognition in Image Classification project for Machine Learning and Deep Learning course's assignment - PoliTO


### Dataset usage - example

* Call the constructor `Cifar100Dataset('split', transform)`
* At each iteration:
    - Call the function `.define_splits(seed)`
    - Call `.change_subclasses(iteration)` to select the list of 10 classes
    - Call `.get_imgs_by_target()` to retrieve all the images belonging to the 10 classes above, ready to fill the pytorch `Subset`.
