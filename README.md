<img src="https://iccs.cam.ac.uk/sites/iccs.cam.ac.uk/files/logo2_1.png"  width="355" align="left">

<br><br><br><br><br>

# ICCS Practical Machine Learning with PyTorch

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/ml-training-material)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This repository contains documentation, resources, and code for the Introduction to
Machine Learning with PyTorch session designed and delivered by Jack Atkinson ([**@jatkinson1000**](https://github.com/jatkinson1000))
and Jim Denholm ([**@jdenholm**](https://github.com/jdenholm)) of [ICCS](https://github.com/Cambridge-ICCS).  
The material has been delivered at both the [ICCS](https://iccs.cam.ac.uk/events/iccs-summer-school-2023) 
and [NCAS](https://ncas.ac.uk/study-with-us/climate-modelling-summer-school/) summer schools.  
All materials, including slides and videos, are available such that individuals can cover the course in their own time.


## Contents

- [Preparation and prerequisites](#preparation-and-prerequisites)
- [Installation and setup](#installation-and-setup)
- [Teaching material](#teaching-material)
- [License information](#license)


## Preparation and prerequisites

To get the most out of the session we assume a basic understanding in a few areas and 
for you to do some preparation in advance.
Expected knowledge is outlined below, along with resources for reading if you are unfamiliar.


### Mathematics and Machine Learning

Basic mathematics knowledge:
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
- Basic mathematical operations
- Writing and running scripts/programs
- Writing and using functions
- The concept of [object orientation](https://eli5.gg/Object-oriented%20programming) (that an object, e.g. a dataset, can have associated functions/methods associated with it.)
- Basic use of the following libraries:
  - [`numpy`](https://numpy.org/) for mathematical and array operations
  - [`matplotlib`](https://matplotlib.org/) for ploting and visualisation
  - [`pandas`](https://pandas.pydata.org/docs/getting_started/index.html) for storing and accessing tabular data
- Familiarity with the [concept of a jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html)

### git and GitHub
You will be expected to know how to
- clone and/or fork a repository,
- commit, and
- push.

The [workshop from the 2022 ICCS Summer School](https://www.youtube.com/watch?v=ZrwzK4CnJ3Q) 
should provide the neccessary knowledge.

### Preparation
In preparation for the course please ensure that your computer contains the following:
- A text editor - e.g. vim/[neovim](https://neovim.io/), [gedit](https://gedit.en.softonic.com/), [vscode](https://code.visualstudio.com/), [sublimetext](https://www.sublimetext.com/) etc. to open and edit code files
- A terminal emulator - e.g. [GNOME Terminal](https://help.gnome.org/users/gnome-terminal/stable/), [kitty](https://sw.kovidgoyal.net/kitty/), [Windows Terminal (windows only)](https://learn.microsoft.com/en-us/windows/terminal/), [iTerm (mac only)](https://iterm2.com/)
- python virtual environment (see [Installation and setup](#installation-and-setup))

Note for Windows users: _We have linked suitable applications for windows in the above lists.
However, you may wish to refer to [Windows' getting-started with python information](https://learn.microsoft.com/en-us/windows/python/beginners)
for a complete guide to getting set up on a Windows system._

If you require assistance or further information with any of these please reach out to
us before a training sesison.


## Installation and setup

### 1. Clone or fork the repository
Navigate to the location you want to install this repository on your system and clone
via https by running:
```
git clone https://github.com/Cambridge-ICCS/ml-training-material.git
```
This will create a directory `ml-training-material/` with the contents of this repository.

Please note that if you have a GitHub account and want to push any work you might do back
up for future reference we suggest you [fork the repository](https://github.com/Cambridge-ICCS/ml-training-material/fork) 
and then clone your fork.


### 2. Create a virtual environment
Before installing any Python packages it is important to first create a Python virtual environment.
This provides an insulated environment inside which we can install Python packages 
without polluting the operating systems's Python environment.

If you have never done this before don't worry: it is *very* good practise, especially 
when you are working on multiple projects, and easy to do.

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

### 4. Run the notebook

From the current directory, launch the Jupyter notebook server:
```
jupyter notebook
```
This command should then point you to the right location within your browser to use the notebook, typically [http://localhost:8888/](http://localhost:8888/).

### (Optional) Keep virtual environment persistent in Jupyter Notebooks

The following step is sometimes useful if you're having trouble with your Jupyter notebook finding the virtual environment. You will want to do this before
launching the Jupyter notebook.
```
python -m ipykernel install --user --name=MLvenv
```

## Teaching Material

### Slides
The slides for this workshop can be viewed on the ICCS Summer School Website:
  - [Teaching](https://cambridge-iccs.github.io/slides/ml-training/slides.html)
  - [Climate Applications](https://cambridge-iccs.github.io/slides/ml-training/applications.html)

The slides are generated from markdown using quarto.
The raw markdown and html files can be found in the [slides](slides/) directory.

### Exercises
The exercises for the course can be found in the [exercises](exercises/) directory.  
These take the form of partially complete jupyter notebooks.

### Videos
Videos from past workshops may be useful if you are following along independently.  
These can be found on the [ICCS youtube channel](https://www.youtube.com/@instituteofcomputingforcli3982) 
under the 2023 Summer School materials.


## License

The code materials in this project are licensed under the [MIT License](LICENSE).

The teaching materials are licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
