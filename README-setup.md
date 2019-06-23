This notebook is the practical component of the [Data Science Summer School](https://www.ds3-datascience-polytechnique.fr) 2019 session on "Learning With Positive Definite Kernels: Theory, Algorithms, and Applications."

It was prepared primarily by [Dougal Sutherland](http://www.gatsby.ucl.ac.uk/~dougals/), based on discussions with [Bharath Sriperumbudur](http://personal.psu.edu/bks18/), and partially based on earlier [materials](https://github.com/karlnapf/ds3_kernel_testing) by [Heiko Strathmann](http://herrstrathmann.de/).

We'll cover, in varying levels of detail, the following topics:

- Solving regression problems with kernel ridge regression:
  - The "standard" approach.
  - Computational/statistical tradeoffs using the Nyström and random Fourier kernel approximations.
  - Learning an appropriate kernel function in a meta-learning setting.
- Two-sample testing with the kernel Maximum Mean Discrepancy (MMD):
  - Estimators for the MMD.
  - Learning an appropriate kernel function.

## Dependencies



### Files
There are a few Python files and some data files in the repository. By far the easiest thing to do is just put them all in the same directory:

```
git clone https://github.com/dougalsutherland/ds3-kernels
```

#### Python version
This notebook requires Python 3.6+. Python 3.0 was released in 2008, and it's time to stop living in the past; most importart Python projects [are dropping support for Python 2 this year](https://python3statement.org/). If you've never used Python 3 before, don't worry! It's almost the same; for the purposes of this notebook, you probably only need to know that you should write `print("hi")` since it's a function call now, and you can write `A @ B` instead of `np.dot(A, B)`.

#### Python packages
We recommend the `conda` package manager; if you don't have it already, install [miniconda](https://docs.conda.io/en/latest/miniconda.html). You can create an environment with everything you need as:

```bash
conda create --name ds3-kernels \
  --override-channels -c conda-forge -c defaults --strict-channel-priority \
  python=3 \
  notebook nb_conda_kernels \
  numpy scipy scikit-learn autograd \
  matplotlib seaborn tqdm

conda activate ds3-kernels

git clone https://github.com/dougalsutherland/ds3-kernels
cd ds3-kernels
jupyter notebook
```

If you have an old conda setup, you can use `source activate` instead of `conda activate`, but it's better to [switch to the new style of activation](https://conda.io/projects/conda/en/latest/release-notes.html#recommended-change-to-enable-conda-in-your-shell). This won't matter for this tutorial, but it's general good practice.

`nb_conda_kernels` makes it easy to switch conda environments inside Jupyter. It's not _necessary_, but it makes life a little easier.

If you don't want to use conda and already have a standard Python 3.6+ and Jupyter setup, we're actually not using anything that out of the ordinary; you can probably get everything not totally ordinary
```
pip install scikit-learn autograd seaborn tqdm
```
All of the imports are right below, so if that runs you shouldn't hit anything later on that'll surprise you.

For the bits that do some "deep learning", you have a few options:

- The default on is [Autograd](https://github.com/HIPS/autograd). Use this if you're not already super-comfortable with one of the other options; it's very easy to use if you already know Numpy. We'll have a brief intro when you need it.
- You can also use PyTorch or TensorFlow (in [eager mode](https://www.tensorflow.org/guide/eager)) if you want. Make the choice by assigning the right value to `engine` in the notebook.
