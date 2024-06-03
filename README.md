<img src="https://socialify.git.ci/sharmachaitanya945/Audio-classification/image?font=Bitter&language=1&name=1&owner=1&pattern=Diagonal%20Stripes&stargazers=1&theme=Dark" alt="Audio-classification" width="640" height="320" />
## Audio Classification with Machine Learning

This project focuses on classifying audio files into different categories using machine learning techniques. The notebook covers the entire pipeline from loading audio data, extracting features, training a neural network model, and making predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to classify audio samples into predefined categories. The project uses the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.

## Dataset

The dataset used in this project is the UrbanSound8K dataset, which can be downloaded from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html). The dataset is divided into 10 classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

## Features

The features used for classification are Mel-Frequency Cepstral Coefficients (MFCCs). These features are extracted from the audio files using the `librosa` library.

## Model Training

The model used in this project is a neural network implemented using TensorFlow and Keras. The model architecture consists of several dense layers with ReLU activation functions and dropout for regularization.

### Training the Model

The model is trained on the extracted MFCC features with the following steps:
1. Load the dataset and preprocess the audio files.
2. Extract MFCC features from the audio files.
3. Split the dataset into training and validation sets.
4. Define the neural network architecture.
5. Compile and train the model, saving the best model based on validation loss.

## Evaluation

The model's performance is evaluated using accuracy and loss metrics on the validation set. The best model is saved during training for later use in making predictions.

## Usage

To use this notebook for your own audio classification tasks, follow these steps:

1. Clone this repository.
2. Download and extract the UrbanSound8K dataset.
3. Install the required dependencies (see below).
4. Run the Jupyter notebook `Audio-classification.ipynb`.

### Predicting New Audio Samples

To classify a new audio sample:
1. Load and preprocess the audio file.
2. Extract MFCC features from the audio file.
3. Load the trained model.
4. Make predictions using the model.

## Requirements

- Python 3.6+
- Jupyter Notebook
- TensorFlow
- Keras
- librosa
- numpy
- scikit-learn

Install the required packages using pip:

```bash
pip install tensorflow keras librosa numpy scikit-learn
```

## Results

The model achieves a validation accuracy of approximately 72%. The performance may vary depending on the specific dataset split and training parameters.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



Feel free to adjust this README to better match your project's specifics.
