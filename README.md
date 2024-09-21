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

<br>
Sure! Here's an updated version of the `README.md` that includes a well-formatted section for the confusion matrix, emphasizing its importance and including the visualization part.

```markdown
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

After training the model, we achieved an accuracy of over 96%. 

### Confusion Matrix

To evaluate the model's performance, we generated a confusion matrix, which provides insights into how well the model is performing on each digit class. Here is the confusion matrix:

```
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
```

### Confusion Matrix Visualization

The confusion matrix can be visualized for better understanding. A heatmap of the confusion matrix was generated using Seaborn:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_true and y_pred are your true labels and predicted labels
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.arange(10), yticklabels=np.arange(10))

plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('Predicted Label', fontsize=15)
plt.ylabel('True Label', fontsize=15)
plt.show()
```

## Installation

To run this project, clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/mnist-handwritten-digit-classification.git
cd mnist-handwritten-digit-classification
pip install -r requirements.txt
```

## Usage

1. **Train the Model**: Run the training script to train the CNN model on the MNIST dataset.
    ```bash
    python src/train.py
    ```

2. **Make Predictions**: Use the trained model to make predictions on test images.
    ```bash
    python src/predict.py
    ```

## License

This project is licensed under the MIT License.
```

### Highlights
- The confusion matrix is clearly introduced, with an explanation of its significance.
- A Python code snippet is provided for visualizing the confusion matrix using a heatmap, making it easy to replicate.
- Ensure to replace `yourusername` in the clone URL with your actual GitHub username.

Feel free to ask for any further changes or details!
