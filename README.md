# Convolutional Neural Network for Pepe Classification

This repository contains code for training a Convolutional Neural Network (CNN) to classify images as "Pepe" or "Not Pepe". The dataset used for training is a synthetic Pepe dataset obtained from the "vikhyatk/synthetic-pepe" dataset available in the Hugging Face's datasets library.

## Libraries Used

- **numpy**: For numerical computations.
- **pandas**: For data manipulation and analysis.
- **tensorflow**: For building and training the CNN model.
- **matplotlib**: For plotting graphs.
- **cv2**: OpenCV library for image processing.

## Dataset

The synthetic Pepe dataset is loaded using the `load_dataset` function from the `datasets` library. The dataset is then split into training, validation, and test sets using a 70-20-10 split ratio.

## Data Preprocessing

- The images are scaled to have pixel values in the range [0, 1].
- The dataset is split into training, validation, and test sets.

## CNN Architecture

The CNN model architecture consists of the following layers:
- Convolutional layer with 16 filters, kernel size (3,3), and ReLU activation.
- MaxPooling layer.
- Convolutional layer with 32 filters, kernel size (3,3), and ReLU activation.
- MaxPooling layer.
- Convolutional layer with 16 filters, kernel size (3,3), and ReLU activation.
- MaxPooling layer.
- Flatten layer.
- Dense layer with 256 neurons and ReLU activation.
- Output layer with 1 neuron and sigmoid activation.

## Training

The model is compiled using the Adam optimizer and binary cross-entropy loss function. It is trained for 10 epochs with training and validation data.

## Evaluation

Model performance is evaluated using precision, recall, and binary accuracy metrics on the test set.

## Testing

The trained model is tested on two sample images to predict whether they belong to the "Pepe" class or "Not Pepe" class.

