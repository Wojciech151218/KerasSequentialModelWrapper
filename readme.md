# Keras Sequential Model Wrapper Library
This small wrapper library simplifies the process of building, training, and using Keras sequential models but could be used for graph models as well.
- Building models based on customizable architecture and training hyperparameters.
- Training models with preconfigured settings, reducing boilerplate code.
- Visualizing individual data points and their predictions.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
    - [Setup and Initialization](#setup-and-initialization)
    - [Defining the Model Architecture](#defining-the-model-architecture)
    - [Training the Model](#training-the-model)
    - [Evaluating Individual Predictions](#evaluating-individual-predictions)

- [Customization](#customization)
- [Examples](#examples)
- [License](#license)

## Installation
This library is built on top of TensorFlow and Keras. To use it, ensure you have the following installed:
- Python >= 3.8
- TensorFlow >= 2.8

You can install TensorFlow using:
``` bash
pip install tensorflow
```
Clone or download the library files into your project directory.
## Features
- **Hyperparameter management**: Easily configure architecture and training hyperparameters (e.g., learning rate, batch size, activation functions, etc.).
- **Model building**: Create models with a custom architecture by subclassing `ModelBuilder`.
- **Training pipeline**: Simplified training process with support for callbacks and hyperparameter-based configurations.
- **Preprocessing utilities**: Supports automatic data preprocessing by overriding `data_funtion` method of `ModelBuilder`.

## Usage
Here’s a step-by-step guide to use the library for defining and training a model.
### Setup and Initialization
Import the required components.
``` python
from hyperparamaters import TrainHyperparameters, ArchitectureHyperparameters
from model_builder import ModelBuilder
import tensorflow as tf
```
# Example on MNIST dataset:

### Defining the Model Architecture
Subclass `ModelBuilder` to define your custom architecture. For example:
``` python
class MyModelBuilder(ModelBuilder):
    def model_building_function(self, callbacks=None) -> tf.keras.Model:
        classes = self.model_trainer.get_classes()

        # Define the model layers
        layers = [
            tf.keras.layers.Flatten(input_shape=self.model_trainer.get_input_shape())
        ]
        for _ in range(self.architecture_hp.layer_count):
            layers.append(
                tf.keras.layers.Dense(
                    self.architecture_hp.neurons_per_layer_count,
                    activation=self.architecture_hp.activation_function
                )
            )
        layers.append(
            tf.keras.layers.Dense(classes, activation="softmax")
        )

        # Return the sequential model
        return tf.keras.models.Sequential(layers)

    @staticmethod
    def data_function(data_function=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (x_train, tf.keras.utils.to_categorical(y_train)), (x_test, tf.keras.utils.to_categorical(y_test))
```
### Training the Model
1. Define architecture hyperparameters:
``` python
architecture_hp = ArchitectureHyperparameters(
    neurons_per_layer_count=50,  # Number of neurons in each layer
    layer_count=5,  # Total number of layers
    activation_function="relu"  # Activation function for hidden layers
)
```
2. Define training hyperparameters:
``` python
train_hp = TrainHyperparameters(
    learning_rate=0.001,  # Learning rate for the optimizer
    epochs=10,  # Number of training epochs
    batch_size=32,  # Size of each training batch
    loss_function="categorical_crossentropy",  # Loss function
    metrics=["accuracy"]  # List of metrics to track
)
```
3. Build the model and start training:
``` python
model_builder = (
    MyModelBuilder()
    .set_hyper_parameters(architecture_hp, train_hp)
    .set_model_trainer()
)
model = model_builder.build_model()
```
### Evaluating Individual Predictions
1. Retrieve the test dataset:
``` python
x_test, y_test = model_builder.get_test_set()
```
2. Evaluate predictions for a single data point:
``` python
sample_index = 0
sample_x = tf.expand_dims(x_test[sample_index], axis=0)  # Add batch dimension
sample_y = tf.expand_dims(y_test[sample_index], axis=0)  # Add batch dimension

prediction = model.predict(sample_x)
print(prediction)

```
## Customization
You can easily extend the library for other datasets and model architectures:
- **Architecture**: Modify `model_building_function()` and `data_function()` in your custom `ModelBuilder` subclass.
- **Training Hyperparameters**: Adjust `TrainHyperparameters` and  `ArchitectureHyperparameters` to suit your training requirements.

## Examples
Check out `example.ipynb` in the repository for a hands-on implementation, including model training, evaluation, and visualization.

## License
This library follows the MIT License. You are free to use, modify, and distribute it as per the terms of the license.
This **README.md** provides a structured guide for users, detailing setup, usage, and customization of the library. Let me know if you'd like further refinements!
