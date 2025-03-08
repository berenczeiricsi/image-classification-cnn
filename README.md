# Image Classification
This project is an example ML project and contains a CNN that is trained to predict the class of an image out of 20 classes.
The input images used are images, which have been converted to grayscale and resized to 100 pixels.

### Architecture
In the `architecture.py` file the Convolutional Neural Network can be found. It has four convolutional blocks, each comprising two convolutional layers with batch normalization, ReLU activation, max pooling, and dropout to process image features progressively.
The classifier block performs adaptive average pooling to a 1x1 spatial dimension, flattens the output, passes it through a fully connected layer, normalizes the output using batch normalization, 
applies a ReLU activation function, uses dropout to prevent overfitting, and finally maps the data to the desired output shape through another fully connected layer.

### Main Script
In the `main.py`file the dataset is loaded once without and once with augmentation. Four sets are created with data loaders for all of them. A separate training set with augmentation is created only for training.
For training and evaluation, the `CrossEntropyLoss` function is used with the `AdamW` optimizer and with learning rate scheduler for adjusting the learning rate during training to optimize model convergence.

The model is trained for a number of epochs, and in each epoch the model is validated on the validation set. 

After the training is finished, a final evaluation is conducted on all sets to test the generalization of the model.

### Example usage
1. Add or download images to a `training_data` folder.
2. Open up a terminal and navigate to the project directory. Then run the following line:

```
python main.py working_config.json
```

### Structure
```
example_project
|- architectures.py
|    Classes and functions for network architectures.
|- datasets.py
|    Dataset classes and dataset helper functions.
|- main.py
|    Main file. In this case also includes training and evaluation routines.
|- README.md
|    A readme file containing info on project, example usage, authors, publication references, and dependencies.
|- utils.py
|    Utility functions and classes. In this case contains a plotting function, an evaluation function and a function 
|    for setting the seed for reporducibility.
|- working_config.json
|     An example configuration file. Can also be done via command line arguments to main.py.
```

### Dependencies
This project was tested on the following dependencies:

- **Python**: Version 3.11
- **Libraries**:
  - `numpy` (1.26.4)
  - `pytorch` (2.2.2)
  - `torchvision` (2.2.2)
  - `matplotlib`(3.8.0)
  - `tqdm` (4.65.0)