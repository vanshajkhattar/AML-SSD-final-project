# 1. SSD Object Detection Algorithm Overview
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


# 2. Research Results Summary of the SSD research paper
## A. PASCAL VOC2007 Dataset Test Results
- SSD object detection algorithm is tested on PASCAL VOC 2007 daataset(4952 images). These results are comparaed against two famous object detection algorithms i,e, Fast R-CNN and Faster R-CNN.


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/vanshajkhattar/AML_SSD_final_project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
