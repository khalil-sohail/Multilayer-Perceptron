# Multilayer-Perceptron

## Description
This repository contains an implementation of a Multilayer Perceptron (MLP) neural network from scratch using Python and NumPy. The MLP is a type of feedforward artificial neural network that consists of multiple layers of neurons, allowing it to learn complex patterns in data.

## Project Structure
```
Multilayer-Perceptron/
├── nn/
│   ├── layers/
│   │   └── dense.py
│   ├── activations/
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── tanh.py
│   │   ├── softmax.py
│   │   └── linear.py
│   ├── losses/
│   │   ├── binary_cross_entropy.py
│   │   ├── categorical_cross_entropy.py
│   │   ├── mse.py
│   │   └── huber.py
│   ├── optimizers/
│   │   ├── sgd.py
│   │   └── adam.py
│   ├── initializers/
│   │   ├── random.py
│   │   ├── xavier.py
│   │   └── he.py
│   ├── metrics/
│   │   ├── accuracy.py
│   │   └── ...
│   ├── callbacks/
│   │   └── early_stopping.py
│   ├── model.py
│   └── serialization.py
│
├── application/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── breast_cancer.py
│   ├── visualization.py
│   └── cli.py
│
├── data/
│   └── data.csv
│
├── tests/
│
└── main.py
```
