# MNIST Handwritten Digit Classification

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model is built using TensorFlow/Keras and provides a strong baseline for digit recognition tasks.

## Libraries Used

- **TensorFlow/Keras**: For building and training the neural network.
- **NumPy**: For numerical operations and data manipulation.
- **Pandas**: For data handling and manipulation.
- **Matplotlib**: For visualizing the training process and results.
- **Seaborn**: For generating a visually appealing confusion matrix.

## Model Architecture

The CNN model architecture consists of:

1. **Input Layer**: Flattening the input images (28x28 pixels).
2. **Hidden Layer 1**: 128 neurons with ReLU activation.
3. **Hidden Layer 2**: 64 neurons with ReLU activation.
4. **Output Layer**: 10 neurons (one for each digit), with softmax activation.

## Training and Results

After training the model, we achieved an accuracy of over 96%. Below is the confusion matrix generated from the model predictions:

[[ 965    1    0    1    0    1    7    4    1    0]
 [   0 1127    1    4    0    1    0    0    2    0]
 [   5    4  996    7    1    0    4    7    6    2]
 [   1    1    5  978    1    5    0    4    5   10]
 [   1    0    0    1  961    0    7    2    0   10]
 [   3    0    0    7    0  866    4    2    3    7]
 [   1    2    0    1    0    2  951    0    1    0]
 [   1    3   10    3    2    0    1  991    2   15]
 [   1    1    3    5    4    4    5    2  937   12]
 [   0    2    0    1   15    3    1    2    1  984]]


