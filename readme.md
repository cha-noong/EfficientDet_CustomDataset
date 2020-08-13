# EfficientDet + CustomDataset + albumentations

This project is customizing the [effiecientdet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) of the Pytorch version.

Dataloader was modified to suit custom data(such as necklace...), and [Data Augmentation library](https://github.com/albumentations-team/albumentations) was also applied.



## Prerequisite

It is highly recommended to install with anaconda
Please refer to the two links
[first link](https://github.com/albumentations-team/albumentations)
[second link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

* Python 3.6 (numpy, skimage, scipy)

* pyotorch >= 1.3

* torchvision >= 0.5.0

* opencv > 4.1

* [albumentations](https://github.com/albumentations-team/albumentations)


GPU is highly recommended. The run time is ~0.01s with GPU(GeForce GTX 1080) and ~0.2s with CPU(Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz).


## Training

Training EfficientDet is a painful and time-consuming task. You shouldn't expect to get a good result within a day or two. Please be patient.


### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
               - image/
                 -*.png
               - annotation/
                 -*.txt
            -val_set_name/
                 - image/
                 -*.png
               - annotation/
                 -*.txt
      
    # for example, coco2017
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val2017/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train2017.json
                -instances_val2017.json

### 2. Manual set project's specific parameters

    # create a yml file {your_project_name}.yml under 'projects'folder 
    # modify it following 'coco.yml'
     
    # for example
    project_name: necklace
    train_set: train
    val_set: val
    num_gpus: 1  # 0 means using cpu, 1-N means using gpus 
    
    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    
    # objects from all labels from your dataset with the order from your annotations.
    # its index must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list: ['necklace']


### 3. Train a custom dataset from scratch

    # train efficientdet-d0 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    for coco
        python train.py -c 0 -p your_project_name --batch_size 8 --lr 1e-5
    
    for custom dataset
        python train_necklace.py
        
    for custom dataset + album
        python train_necklace_album.py



 
