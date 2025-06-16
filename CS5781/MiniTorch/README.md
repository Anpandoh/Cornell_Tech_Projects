# MiniTorch Recreation

This repository contains my implementation of the [MiniTorch](https://minitorch.github.io/) project — a pedagogical re-creation of the core components of PyTorch. MiniTorch is designed as a teaching framework to help understand how deep learning libraries like PyTorch work under the hood.

## About MiniTorch

MiniTorch is a series of programming assignments developed to guide learners through building a small but functional subset of the PyTorch framework. Each module introduces a new concept in numerical computing and deep learning, allowing you to implement and understand these systems step by step.

The official MiniTorch project can be found here: [https://minitorch.github.io/](https://minitorch.github.io/)

---

## My Implementation

This repository is structured by modules, with each module building upon the previous one. Below is a summary of what each module contains:

### 🔹 Module 0 – Basics
This module focuses on foundational Python and numerical programming concepts, including:
- Scalars, operators, and basic mathematical functions
- Broadcasting and manual computation
- Introduction to structured testing and code design

### 🔹 Module 1 – Autodifferentiation
In this module, I implemented reverse-mode automatic differentiation, which is the backbone of gradient-based optimization. Topics include:
- Computation graphs
- Backpropagation
- Gradients of scalar functions

### 🔹 Module 2 – Tensors
Here, I created a basic tensor class with support for:
- N-dimensional arrays
- Broadcasting, reshaping, and element-wise operations
- Gradient tracking through tensor operations

### 🔹 Module 3 – Parallelization (CUDA/Numba)
This module introduces performance improvements via parallel computation:
- Custom kernels with Numba
- Device abstraction to support CPU and (optionally) GPU computation
- Efficient implementations of tensor operations

### 🔹 Module 4 – Neural Networks
The final module brings everything together to build neural networks:
- Layers and models
- Loss functions and optimizers
- Training loop and evaluation

---

## Goals

- Deepen understanding of how deep learning libraries are structured
- Learn the inner mechanics of tensors, autograd, and neural networks
- Gain practical experience with performance optimization tools like Numba

---

## Getting Started

To run the code in this repository, you'll need Python 3.8+ and the following dependencies:
- `numpy`
- `numba`
- `pytest`

You can install the requirements with:

```bash
pip install -r requirements.txt
