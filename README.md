# PyTorch-DEEP-LEARNING

## PyTorch - DEEP LEARNING WITH PYTORCH
### Using MNIST Datasets
PyTorch is an open-source machine learning library for Python, based on Torch, used for applications such as natural language processing. It is primarily developed by Facebook's artificial-intelligence research group, and Uber's "Pyro" software for probabilistic programming is built on it.

### The MNIST dataset

The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.

The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts: - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples) - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels) - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples) - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)

### PyTorch provides two high-level features:

a) Tensor computation (like NumPy) with strong GPU acceleration

b) Deep Neural Networks built on a tape-based autodiff system

To keep things short:

### PyTorch consists of 4 main packages:

torch: a general purpose array library similar to Numpy that can do computations on GPU when the tensor type is cast to (torch.cuda.TensorFloat)

torch.autograd: a package for building a computational graph and automatically obtaining gradients

torch.nn: a neural net library with common layers and cost functions

torch.optim: an optimization package with common optimization algorithms like SGD,Adam, etc

### PyTorch Tensors

In terms of programming, Tensors can simply be considered multidimensional arrays. Tensors in PyTorch are similar to NumPy arrays, with the addition being that Tensors can also be used on a GPU that supports CUDA. PyTorch supports various types of Tensors.

### References:

PyTorch:  https://pytorch.org/

Example with MNIST Datasets:  https://gist.github.com/reddragon/3fa9c3ee4d10a7be242183d2e98cfc5d

Git Hsaghir: https://hsaghir.github.io/data_science/pytorch_starter/

