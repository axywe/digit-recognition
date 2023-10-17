# Simple Neural Network for Digit Recognition

## Overview
This repository contains Python code for a simple feedforward neural network (NN) designed to classify images. The network is implemented from scratch and can be configured to have a varying number of neurons in its hidden layer. The project includes a Flask web interface for interactive image classification.

## Features
- Customizable hidden layer size
- Performance metrics including accuracy, F1 score, precision, and recall
- Test script to evaluate performance on different hidden layer sizes
- Flask web interface for real-time image classification
- Weight saving for trained models
- Training and evaluation scripts

## Dependencies
- Python 3.x
- NumPy
- scikit-learn
- PIL (Pillow)
- Flask

## File Descriptions
- `simple_nn.py`: Contains the SimpleNN class, which includes methods for both forward and backward passes.
- `train_model.py`: Script for training the neural network model.
- `test_script.py`: Script for testing the model's performance with different hidden layer sizes.
- `main.py`: Flask web application script for real-time image classification.

## Setup
1. Clone this repository.
2. Install the required packages.
3. **Important**: Download the MNIST dataset from [this GitHub repository](https://github.com/pjreddie/mnist-csv-png) and place the PNG images in a folder named `img` for training.
4. Run `train_model.py` to train the neural network model. This will also save the weights to the `weights/` folder.
5. Run `test_script.py` to evaluate the model's performance with different hidden layer sizes. The results will be saved in `new_objective_test_results.json`.
6. Run `main.py` to start the Flask web application.

## Usage
- Run `train_model.py` to train a new model and save its weights.
- Run `test_script.py` to evaluate the model using saved weights.
- Open your browser and navigate to `http://127.0.0.1:5000/` to interact with the Flask web interface.

## Metrics
The performance of the model is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified instances.
- **F1 Score**: The harmonic mean of precision and recall.
- **Precision**: The number of true positives divided by the number of true and false positives.
- **Recall**: The number of true positives divided by the number of true positives and false negatives.