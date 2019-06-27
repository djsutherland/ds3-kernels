This is the practical component of the [Data Science Summer School](https://www.ds3-datascience-polytechnique.fr) 2019 session on "Learning With Positive Definite Kernels: Theory, Algorithms, and Applications."
Slides are available [here](https://github.com/dougalsutherland/ds3-kernels/tree/slides).

It was prepared primarily by [Dougal Sutherland](http://www.gatsby.ucl.ac.uk/~dougals/), based on discussions with [Bharath Sriperumbudur](http://personal.psu.edu/bks18/), and partially based on earlier [materials](https://github.com/karlnapf/ds3_kernel_testing) by [Heiko Strathmann](http://herrstrathmann.de/).

We'll cover, in varying levels of detail, the following topics:

- Solving regression problems with kernel ridge regression ([`ridge.ipynb`](ridge.ipynb)):
  - The "standard" approach.
  - Computational/statistical tradeoffs using the Nyström and random Fourier kernel approximations.
  - Learning an appropriate kernel function in a meta-learning setting.
- Two-sample testing with the kernel Maximum Mean Discrepancy (MMD) ([`testing.ipynb`](testing.ipynb)):
  - Estimators for the MMD.
  - Learning an appropriate kernel function.

## Dependencies

### Colab

These notebooks are available on Google Colab: [ridge](https://colab.research.google.com/github/dougalsutherland/ds3-kernels/blob/built/ridge.ipynb) or [testing](https://colab.research.google.com/github/dougalsutherland/ds3-kernels/blob/built/testing.ipynb). You don't have to set anything up yourself and it runs on cloud resources, so this is probably the easiest option. If you want to use the GPU, click Runtime -> Change runtime type -> Hardware accelerator -> GPU.

### Local setup

Run `check_imports_and_download.py` to see if everything you need is installed (and download some more small datasets if necessary). If that works, you're set; otherwise, read on.


### Files
There are a few Python files and some data files in the repository. By far the easiest thing to do is just put them all in the same directory:

```
git clone https://github.com/dougalsutherland/ds3-kernels
```

#### Python version
This notebook requires Python 3.6+. Python 3.0 was released in 2008, and it's time to stop living in the past; most importart Python projects [are dropping support for Python 2 this year](https://python3statement.org/). If you've never used Python 3 before, don't worry! It's almost the same; for the purposes of this notebook, you probably only need to know that you should write `print("hi")` since it's a function call now, and you can write `A @ B` instead of `np.dot(A, B)`.

#### Python packages

The main thing we use is PyTorch and Jupyter. If you already have those set up, you should be fine; just additionally make sure you also have (with `conda install` or `pip install`) `seaborn`, `tqdm`, and `sckit-learn`. We import everything right at the start, so if that runs you shouldn't hit any surprises later on.

If you don't already have a setup you're happy with, we recommend the `conda` package manager - start by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then you can create an environment with everything you need as:

```bash
conda create --name ds3-kernels \
  --override-channels -c pytorch -c defaults --strict-channel-priority \
  python=3 notebook ipywidgets \
  numpy scipy scikit-learn \
  pytorch=1.1 torchvision \
  matplotlib seaborn tqdm

conda activate ds3-kernels

git clone https://github.com/dougalsutherland/ds3-kernels
cd ds3-kernels
python check_imports_and_download.py
jupyter notebook
```

(If you have an old conda setup, you can use `source activate` instead of `conda activate`, but it's better to [switch to the new style of activation](https://conda.io/projects/conda/en/latest/release-notes.html#recommended-change-to-enable-conda-in-your-shell). This won't matter for this tutorial, but it's general good practice.)

(You can make your life easier when using jupyter notebooks with multiple kernels by installing `nb_conda_kernels`, but as long as you install and run `jupyter` from inside the env it will also be fine.)


## PyTorch

We're going to use PyTorch in this tutorial, even though we're not doing a ton of "deep learning." (The CPU version will be fine, though a GPU might let you get slightly better performance in some of the "advanced" sections.)

If you haven't used PyTorch before, don't worry! The API is unfortunately a little different from NumPy (and TensorFlow), but it's pretty easy to get used to; you can refer to [a cheat sheet vs NumPy](https://github.com/wkentaro/pytorch-for-numpy-users/blob/master/README.md) as well as the docs: [tensor methods](https://pytorch.org/docs/stable/tensors.html) and [the `torch` namespace](https://pytorch.org/docs/stable/torch.html#torch.eq). Feel free to ask if you have trouble figuring something out.
