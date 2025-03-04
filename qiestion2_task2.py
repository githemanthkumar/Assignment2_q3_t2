import tensorflow as tf
from tensorflow.keras import layers, models

# Define the residual block function
def residual_block(input_tensor, filters):
    # First Conv2D layer
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    
    # Second Conv2D layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)

    # Skip connection: Add input_tensor to the output of the second Conv2D layer
    x = layers.add([x, input_tensor])

    # Apply activation (ReLU) after adding the skip connection
    x = layers.Activation('relu')(x)

    return x

# Build the ResNet-like model
def resnet_model():
    input_layer = layers.Input(shape=(224, 224, 3))  # Input shape for the model (ImageNet standard size)

    # Initial Conv2D layer with 64 filters, kernel size = (7,7), and stride = 2
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)  # MaxPooling after the initial Conv2D

    # Apply two residual blocks with 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Flatten layer
    x = layers.Flatten()(x)

    # Fully connected Dense layer with 128 neurons and ReLU activation
    x = layers.Dense(128, activation='relu')(x)

    # Output layer with 10 neurons and softmax activation (for multi-class classification)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the ResNet-like model
model = resnet_model()

# Print the model summary
model.summary()
