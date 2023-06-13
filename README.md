# ICCS PyTorch training

This repository contains documentation, resources, and code for the Introduction to
Machine Learning with PyTorch session deliovered by @jatkinson1000 and @jdenholm of
[ICCS](https://github.com/Cambridge-ICCS) at the [2023 summer school](https://iccs.cam.ac.uk/events/iccs-summer-school-2023).


## Contents
[Installation and Setup](#installation-and-setup)


## Installation and setup

### 1. Clone the repository
Navigate to the location you want to install this repository on your system and clone
via https by running:
```
git clone https://github.com/Cambridge-ICCS/ml-training-material.git
```
This will create a directory `ml-training-material/` with the contents of this repository.


### 2. Create a virtual environment
Before installing any Python packages it is important to first create a Python virtual environment.
This provides an insulated environment inside which we can install Python packages without polluting the operating systems's Python environment.

If you have never done this before don't worry: it is *very* good practise, especially when you are working on multiple projects, and easy to do.

```
python3 -m venv MLvenv
```
This will create a directory called `MLvenv` containing software for the virtual environment.
To activate the environment run:
```
source MLvenv/bin/activate
```
You can now work on python from within this isolated environment, installing packages
as you wish without disturbing your base system environment.

When you have finished working on this project run:
```
deactivate
```
to deactivate the venv and return to the system python environment.

You can always boot back into the venv as you left it by running the activate command again.


### 3. Install dependencies

It is now time to install the dependencies for our code, for example PyTorch.
The project has been packaged with a [`pyproject.toml`](pyproject.toml) so can be installed in one go.
From within the root directory in a active virtual environment run:
```
pip install .
```
This will download the relevant dependencies into the venv as well as setting up the datasets that we will be using in the course.
