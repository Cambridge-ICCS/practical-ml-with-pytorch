# Slides

The slides for this course can be viewed on the ICCS Website:
  - [Teaching](https://cambridge-iccs.github.io/slides/ml-training/slides.html)
  - [Climate Applications](https://cambridge-iccs.github.io/slides/ml-training/applications.html)

This directory contains the source for the slides.
They have been written in [Quarto](https://quarto.org/) markdown and rendered to html
using quarto-cli.

This can be done navigating into the `slides/` directory and running:
```
quarto render slides.qmd
quarto render applications.qmd
```
with quarto installed.

## Part 1 - ML and NN basics

The first part of the theory slides deals with:
- The concepts of regression
  - Gradient descent
  - The structure of a NN - just the same but bigger
- Uses of Neural Nets
  - Classification and regression
    - differences
    - applications
- PyTorch and basic python structure of a project
- Penguin examples
  - 1 classification, 1 regression
  - Fully Connected Neural Networks
    - How to train and run inference
  - Deeper theoretical details
    - Relu Layers, Activation functions, when to/not to use
    - How is data represented as a tensor
    - Selecting an optimiser
    - Structuring a pytorch NN code
    - Multiple outputs
  - How to make use of PyTorch Docs?
    - Where to go for more information

## Day 2 - CNNs and more

The second part of the theory slides deals with Convolutional Neural Networks:
- What is a CNN?
- How is a CNN different from NNs?
  - Encoders
  - Where to use?
    - Image-like problems
    - Segmentation/classification/regression?

## Part 3 - Climate Applications

The final part of the course, in the applications slides, deals with various potential
ways in which machine learning might be used within geosciences along with considrations,
pitfalls, and sources for further reading.
