# Lab 1: Environment Setup and Signal Acquisition

Please start here for instructions on how to complete lab1!

## Setup

### Install Conda
Conda is an open-source, cross-platform tool used to manage software packages and virtual environments. 

Install Miniconda or another Conda distribution by following the [official Conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/). After installation, open a new terminal and verify that Conda is available:

```bash
# check Conda has been installed successfully
conda --version
```

### Python and Conda Environment
We will use Conda throughout the software labs to manage our package environment. To create and activate the course environment, run:

```bash
# make sure you are in the repository containing all the lab assignments
conda create -n ee194-21 python=3.11
conda activate ee194-21 # activate the conda environment
```

Every time you open a new terminal, remember to activate the conda environment again with `conda activate ee194-21`. Next, install the lab package:

```bash
cd lab1
pip install -e . # this might take a while
```

### Jupyter and Jupyter notebook

Jupyter is an interactive computing environment commonly used for scientific programming, data analysis, and visualization. A Jupyter Notebook combines executable Python code, formatted text, equations, figures, and results in a single document. 

Throughout the software labs, the main deliverables will be the `lab*.ipynb` jupyter notebook. If you are new to jupyter notebook or jupyter lab, please feel free to check the tutorial [here](https://jupyter.org/try-jupyter/notebooks/?path=notebooks%2FIntro.ipynb)

Install Jupyter:

```bash
conda install jupyter
```

From the `lab1` directory, run:

```bash
jupyter lab
```

Open `lab1.ipynb` and complete each section.

## Completing the notebook

Please now go over and follow the instructions in the notebook `lab1.ipynb`. 
- Complete code in every block that has "TODO:"
- Answer all the questions

## Submission

Finally, submit both `lab1.ipynb` and the Rendered PDF of the notebook to Gradescope.