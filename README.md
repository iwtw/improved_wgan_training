# Face images super-resolution using improved_wgan_training

forked from igul222/improved_wgan_training 
["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028)
## Prerequisites
- Python, NumPy, TensorFlow >= r1.0, SciPy, Matplotlib
## Models
discriminator receive inputs of size [112,96,3] face images (e.g. webface dataset) and generator receive [ 28 , 24 , 3 ]

using residualblock for both discriminator and generator

using a generator from SRGAN generator

using WGAN cost

replace deconv layer with pixelshuffle layer for performance

