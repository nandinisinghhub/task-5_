Neural Style Transfer using TensorFlow
This notebook demonstrates how to perform Neural Style Transfer using TensorFlow. The code implements the technique described in the paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

Neural Style Transfer is an optimization technique used to combine the content of one image with the style of another image.

Requirements
The code requires the following libraries:

tensorflow
tensorflow_hub
matplotlib
numpy
PIL (Pillow)
These libraries are imported at the beginning of the notebook.

Code Overview
The notebook consists of several code cells that perform the following steps:

Import necessary libraries: Imports TensorFlow, TensorFlow Hub, Matplotlib, NumPy, PIL, and other utilities.
Define helper functions:
tensor_to_image: Converts a TensorFlow tensor to a PIL Image.
load_img: Loads and preprocesses an image for the model. It resizes the image while maintaining the aspect ratio.
imshow: Displays an image using Matplotlib.
Load content and style images: Downloads example content and style images and loads them using the load_img function.
Visualize content and style images: Displays the loaded content and style images.
Load pre-trained VGG19 model: Loads the VGG19 model from TensorFlow Hub. This model is used to extract features from the images. An alternative approach using tf.keras.applications.VGG19 is also shown for feature extraction.
Define content and style layers: Specifies the layers from the VGG19 model that will be used to extract content and style features.
Create a model for extracting intermediate layers: Defines a function vgg_layers that creates a Keras model returning the outputs of the specified intermediate layers of the VGG19 model.
Calculate Gram Matrix: Defines a function gram_matrix to compute the Gram matrix, which is used to represent the style of an image.
Define the StyleContentModel: Creates a custom Keras Model StyleContentModel that takes content and style images as input and returns the style and content representations from the chosen layers.
Set up style and content targets: Extracts the style features from the style image and content features from the content image using the StyleContentModel.
Initialize the image to be optimized: Creates a TensorFlow Variable initialized with the content image. This image will be iteratively updated during the optimization process.
Define optimization parameters: Sets up the Adam optimizer and defines weights for the style and content losses.
Define loss functions:
style_content_loss: Calculates the total loss based on the style and content losses.
total_variation_loss: Calculates the total variation loss to encourage spatial smoothness in the output image.
Define the training step: Creates a @tf.function decorated train_step that performs a single optimization step. It calculates the loss, computes gradients, applies gradients using the optimizer, and clips the image pixel values to the range [0, 1].
Run the optimization: Iteratively runs the train_step for a specified number of epochs and steps per epoch to transfer the style. The styled image is displayed periodically during training.
Visualize image deltas and Sobel edges: Shows how the styled image differs from the original content image in terms of horizontal and vertical changes.
Calculate total variation loss: Demonstrates how to calculate the total variation loss for the styled image.
Save the styled image: Saves the final stylized image to a file and attempts to download it in a Colab environment.
Usage
Run all the code cells in the notebook sequentially.
Adjust the style_weight, content_weight, and total_variation_weight to control the trade-off between preserving content, transferring style, and image smoothness.
Modify the content_path and style_path variables to use your own content and style images.
Note
The VGG19 model weights are downloaded automatically when the corresponding cells are executed for the first time.
