# Assignment2_q3_t2
Residual Block:

The residual_block function applies two Conv2D layers with 64 filters and a 3x3 kernel. A skip connection is added where the input tensor is summed with the output from the two convolutional layers. The addition is followed by a ReLU activation.
ResNet Model:

The model starts with an initial Conv2D layer with 64 filters, a 7x7 kernel, and a stride of 2, followed by max-pooling.
Two residual blocks are applied with 64 filters each. These blocks leverage skip connections to preserve gradients and improve training.
After the residual blocks, the output is flattened and passed through a dense layer with 128 neurons and ReLU activation.
The output layer has 10 neurons with a softmax activation for multi-class classification.
Model Summary:

The model summary is printed, showing the architecture, output shapes of each layer, and the number of parameters.
Output Example (Model Summary):
bash
Copy
Edit
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 112, 112, 64)      9472
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 56, 56, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 56, 56, 64)        36928
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 56, 56, 64)        36928
_________________________________________________________________
activation (Activation)      (None, 56, 56, 64)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        36928
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 56, 56, 64)        36928
_________________________________________________________________
activation_1 (Activation)    (None, 56, 56, 64)        0
_________________________________________________________________
flatten (Flatten)            (None, 200704)            0
_________________________________________________________________
dense (Dense)                (None, 128)               25690240
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 25,844,714
Trainable params: 25,844,714
Non-trainable params: 0
_________________________________________________________________
This script defines a simplified ResNet-like model with residual blocks and ends with fully connected layers for classification tasks.
