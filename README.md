# ICCS PyTorch training

This repository contains documentation, resources, and code for the Introduction to
Machine Learning with PyTorch session deliovered by [**@jatkinson1000**](https://github.com/jatkinson1000)
and [**@jdenholm**](https://github.com/jdenholm) of
[ICCS](https://github.com/Cambridge-ICCS) at the [2023 summer school](https://iccs.cam.ac.uk/events/iccs-summer-school-2023).


## Contents
[Preparation and prerequisites](#preparation-and-prerequisites)
[Installation and setup](#installation-and-setup)
[Teaching material](#teaching-material)


## Preparation and prerequisites

To get the most out of the session we expact a basic understanding in a few areas and for you to do some preparation.
Topics expected are outlined below, along with some resources for reading if you are unfamiliar.

### Mathematics and Machine Learning

Basic knowledge of:
- calculus - differentiating a function
- matrix algebra - matrix multiplication and representing data as a matrix
- regression - fitting a function to data

Neural Networks:
- Awareness of high-level concepts 
- We recommend the [video series by 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks), at least chapters 1-3.

### Python
The course will be taught in python using pyTorch.
Whilst no prior knowledge of pyTorch is expected we assume users are familiar with the basics of Python3.
This includes:
- Writing and using functions
- Basic mathematical operations
- Writing and running scripts/programs
- Basic use of the following libraries:
  - numpy for mathematical and array operations
  - matplotlib for ploting and visualisation
  - pandas for storing and accessing data
- Familiarity with the concept of a jupyter notebook

### git and GitHub
The course from day 1 of the summer school should provide the neccessary knowledge.
You will be expected to know how to clone and/or fork a repository, commit, and push.

### Preparation
In preparation for the course please ensure that your computer contains the following:
- Text editor - e.g. vim, gedit, vscode, sublimetext etc. to open and edit code files
- UNIX terminal - e.g. Terminal, iTerm, kitty, 
- python virtual environment (see [Installation and setup](#installation-and-setup))

If you are a windows user we strongly advise working in linux.
The easiest way to do this is [using Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).
Virtual machines or dual-booting are also options.

If you require assistance or further information with any of these please reach out to us before the school.


## Installation and setup

### 1. Clone or fork the repository
Navigate to the location you want to install this repository on your system and clone
via https by running:
```
git clone https://github.com/Cambridge-ICCS/ml-training-material.git
```
This will create a directory `ml-training-material/` with the contents of this repository.

Please note that if you have a GitHub account and want to push any work you might do back up for future reference we 
suggest you [fork the repository](https://github.com/Cambridge-ICCS/ml-training-material/fork) and then clone your fork.


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
These are all listed in the [`requirements.txt`](requirements.txt) file and can be installed in one go.
From within the root directory in a active virtual environment run:
```
pip install -r requirements.txt
```

## Teaching Material

### Slides
The slides for this workshop are generated from markdown using quarto.
The raw markdown and html files can be found in the [slides](#/slides/) directory.
The slides can be viewed at [TODO: Add link once ready]()

### Exercises
The exercises that we will be working on can be found in the [exercises](#/exercises/) directory.
These take the form of jupyter notebooks.


