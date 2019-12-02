# SSD Object Detection Algorithm Overview
## A. Concepts
- __Object Detection__ Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos.
- __Single Shot Detection__ Single-shot models encapsulate both localization and detection tasks in a single forward sweep of the network, resulting in significantly faster detections while deployable on lighter hardware.
- __Multiscale Feature Maps__ In object detection, feature maps from intermediate convolutional layers can be directly useful because they represent the original image at different scales. Therefore, a fixed-size filter operating on different feature maps will be able to detect objects of various sizes.
- __Priors__ These are pre-computed boxes defined at specific positions on specific feature maps, with specific aspect ratios and scales. 
- __Multibox__ This is a technique that formulates predicting an object's bounding box as a regression problem, wherein a detected object's coordinates are regressed to its ground truth's coordinates. In addition, for each predicted box, scores are generated for various object types.
- __Hard Negative Mining__  This refers to explicitly choosing the most egregious false positives predicted by a model and forcing it to learn from these examples.
- __Non-Maximum Suppression__ Non-Maximum Suppression (NMS) is a means to remove redundant predictions by suppressing all but the one with the maximum score.

## B. Overview
The SSD is a purely convolutional neural network (CNN) that we can organize into three parts –

1. Base convolutions derived from an existing image classification architecture that will provide lower-level feature maps.

2. Auxiliary convolutions added on top of the base network that will provide higher-level feature maps.

3. Prediction convolutions that will locate and identify objects in these feature maps.

The paper demonstrates two variants of the model called the SSD300 and the SSD512. The suffixes represent the size of the input image. Although the two networks differ slightly in the way they are constructed, they are in principle the same. The SSD512 is just a larger network and results in marginally better performance.

![SSD_Architecture](https://miro.medium.com/max/974/1*51joMGlhxvftTxGtA4lA7Q.png)

<a href="https://imgflip.com/gif/3i652s"><img src="https://i.imgflip.com/3i652s.gif" title="made at imgflip.com"/></a>


# 1. Results Summary of the SSD Research Paper.
- Experiments in the paper are all based on VGG16, which is pre-trained on the ILSVRC CLS-LOC dataset. 
- SSD object detection algorithm is tested on PASCAL VOC 2007 daataset(4952 images). These results are comparaed against two famous object detection algorithms i,e, Fast R-CNN and Faster R-CNN.


## A. PASCAL VOC2007 Dataset Test Results.


![VOC2007](https://drive.google.com/drive/u/0/my-drive)

## B. PASCAL VOC2012 Dataset Test Results

## C. COCO test-dev2015 detectionresults.
## D. Data Augmentation for Small Object Accuracy Results

# 2.  Procedure taken to Reproduce the Results.(Our Procedure and Results to reproduce the SSD algorithm)

# 3.  Describes measurements and/or analysis of what was discovered when attempting to reproduce result.
# 4.  Discussion and References to relevant papers.
# 5.  References.



