# DCGAN


Minimalistic tf 2.5 implementation of DCGAN with support for distributed training on multiple GPUs. From older ressources:
```
https://github.com/adityabingi/DCGAN-TF2.0
https://github.com/carpedm20/DCGAN-tensorflow
```

This work is aimed to generate novel face images similar to CelebA image dataset using Deep Convolutional Generative Adversarial Networks (DCGAN).

For theory of GAN's and DCGAN refer these works:
1. [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
2. [NIPS 2016 Tutorial:Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
3. [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)


## Code compatibility:
```
python>=3.8
Tensorflow==2.5.0
cuDNN==8.1.0
CUDA==11.4
```

## Setup
```
pip install tensorflow-gpu==2.5
pip install sklearn
pip install image
pip install tqdm
```
## Dataset

`python download.py celebA`

Extract **img_align_celeba.zip** and the images are found in the **celebA** folder.

Data Processing:

All the images in the celeba dataset are of (218 ,178, 3) resolution and for this work all the images are cropped by carefully choosing the common face region (128, 128, 3) in all the images. Check data_crop in config.py

## Usage

For gpu training:

`python dcgan.py --train`

To run on single GPU run the above code by simply setting **num_gpu = 1** in config.py.

For Generating new samples:

`python dcgan.py --generate`


## Results

Following are the results after training GAN on 128x128 resolution CelebA face images for 15 epochs on 2 NVIDIA Tesla K80 GPUs with global batch size of 32 (batch size 16 per gpu). Detailed configuration can be found in config.py 


Fake Images Generation after 30 Epochs:

![training-result](results/dcgan_training.gif)

Fake Images Generation after 30 Epochs:
![results_15epoch](results/fakes_epoch30.jpg)

