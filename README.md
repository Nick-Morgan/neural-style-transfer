# Update - March 2021
- I have created a new repo, ([nst-zoo](https://github.com/Nick-Morgan/nst-zoo))which is much, much more thorough than this one. 
- This repo contains some spaghetti code, but contains some fun images, and will probably stay up forever since it made it into the arctic code vault.



# neural-style-transfer

![Example Image](img/example.png)

Follow along in a GPU-enabled workbook! [Link](https://colab.research.google.com/github/Nick-Morgan/neural-style-transfer/blob/master/Neural_Style_Transfer.ipynb). 

Neural Style Transfer utilizes the [VGG-19 Image Classification Neural Network](https://arxiv.org/pdf/1409.1556.pdf) to apply transfer learning to images. 

![VGG19 - Clifford K. Yang](https://github.com/Nick-Morgan/neural-style-transfer/blob/master/img/vgg19.jpg?raw=1)

This repository explores two methods - one introduced by Leon A. Gatys, and another introduced by Justin Johnson.

[Gatys' method](https://arxiv.org/pdf/1508.06576.pdf) is an iterative process (typically 150-200 iterations) to optimize a generated image, based on a cost function for style and content. The content cost function is defined by comparing the outputs of `conv4_2` between the generated image and the content image. The style cost function is defined by comparing the outputs of `[conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]` between the generated image and the style image.

Instead of using an iterative optimization, [Johnson's method](https://arxiv.org/pdf/1603.08155.pdf) transforms images using a single forward pass through the network. It still uses the pre-trained VGG network, but instead trains the model to make transformations using 80,000 training images from the [Microsoft COCO dataset](https://arxiv.org/pdf/1405.0312.pdf). A single forward pass through this network has a loss which is comparable to ~100 iterations of Gatys' method. 


The main notebook, Neural_Style_Transfer.ipynb, contains all relevant documentation for this repository.
