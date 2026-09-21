# ============================================================
# GRAD-CAM EXPLAINABILITY UTILITY
# ============================================================
# Works with:
#   1. EfficientNetB3 - Brain Tumor
#   2. ResNet50        - Breast Cancer
#
# Designed for Keras 3 / TensorFlow
# ============================================================

import os
import numpy as np
import tensorflow as tf

from PIL import Image


# ============================================================
# 1. FIND THE BASE CNN MODEL
# ============================================================

def find_base_model(model):
    """
    Find a nested CNN model inside the complete model.

    For our project this will normally be:
        EfficientNetB3
        or
        ResNet50
    """

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            return layer

    return model


# ============================================================
# 2. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):
    """
    Find the last convolutional layer inside the CNN.
    """

    # Search from the end of the model
    for layer in reversed(model.layers):

        # Check whether this is a nested model
        if isinstance(layer, tf.keras.Model):

            # Search inside the nested model
            for inner_layer in reversed(
                layer.layers
            ):

                try:

                    output_shape = (
                        inner_layer.output.shape
                    )

                    # A convolutional feature map
                    # normally has 4 dimensions:
                    #
                    # batch, height, width, channels

                    if len(output_shape) == 4:

                        return layer, inner_layer

                except Exception:

                    continue

    # If no nested model was found,
    # search directly in the main model

    for layer in reversed(model.layers):

        try:

            output_shape = (
                layer.output.shape
            )

            if len(output_shape) == 4:

                return model, layer

        except Exception:

            continue

    raise ValueError(
        "Could not find a suitable convolutional layer."
    )


# ============================================================
# 3. LOAD IMAGE
# ============================================================

def load_and_preprocess_image(
    image_path,
    image_size
):
    """
    Load an image and convert it into
    the format used by our models.
    """

    # Check image
    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open image
    image = Image.open(
        image_path
    )

    # Convert to RGB
    image = image.convert(
        "RGB"
    )

    # Resize
    image = image.resize(
        image_size
    )

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize exactly as our
    # prediction scripts do
    image_array = image_array / 255.0

    # Save original normalized image
    original_image = image_array.copy()

    # Add batch dimension
    input_tensor = np.expand_dims(
        image_array,
        axis=0
    )

    # Convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32
    )

    return original_image, input_tensor


# ============================================================
# 4. GENERATE GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(
    image_tensor,
    model,
    target_class_index=None
):
    """
    Generate Grad-CAM heatmap.

    target_class_index:
        None  -> binary model
        0,1,2... -> multiclass model
    """

    # --------------------------------------------------------
    # Find CNN base model and final convolutional layer
    # --------------------------------------------------------

    base_model, last_conv_layer = (
        find_last_conv_layer(model)
    )

    print(
        "Grad-CAM base model:",
        base_model.name
    )

    print(
        "Grad-CAM layer:",
        last_conv_layer.name
    )

    # --------------------------------------------------------
    # Build a model that returns:
    #
    # 1. Last convolutional feature maps
    # 2. Base CNN output
    #
    # This avoids the Keras nested-model connection problem.
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # --------------------------------------------------------
    # Find the layers after the CNN base model
    # --------------------------------------------------------

    base_index = None

    for i, layer in enumerate(
        model.layers
    ):

        if layer is base_model:

            base_index = i

            break

    if base_index is None:

        raise ValueError(
            "Could not locate base model inside main model."
        )

    # --------------------------------------------------------
    # Layers after the CNN
    # --------------------------------------------------------

    classifier_layers = model.layers[
        base_index + 1:
    ]

    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Get feature maps and CNN output
        conv_outputs, base_output = (
            feature_model(
                image_tensor,
                training=False
            )
        )

        # Pass CNN output through the remaining
        # classification layers manually
        predictions = base_output

        for layer in classifier_layers:

            predictions = layer(
                predictions,
                training=False
            )

        # ----------------------------------------------------
        # Binary model
        # ----------------------------------------------------

        if target_class_index is None:

            target = predictions[:, 0]

        # ----------------------------------------------------
        # Multiclass model
        # ----------------------------------------------------

        else:

            target = predictions[
                :,
                target_class_index
            ]

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        target,
        conv_outputs
    )

    # --------------------------------------------------------
    # Average gradients over height and width
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    # --------------------------------------------------------
    # Weight feature maps by gradients
    # --------------------------------------------------------

    weighted_features = (
        conv_outputs
        *
        pooled_gradients
    )

    # Average across channels
    heatmap = tf.reduce_mean(
        weighted_features,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize heatmap
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value
        +
        tf.keras.backend.epsilon()
    )

    # Convert to NumPy
    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# 5. CREATE HEATMAP OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.40
):
    """
    Overlay Grad-CAM heatmap on the original image.
    """

    # Convert original image from 0-1 to 0-255
    original_uint8 = np.uint8(
        np.clip(
            original_image * 255,
            0,
            255
        )
    )

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap * 255,
            0,
            255
        )
    )

    # Convert heatmap to PIL image
    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    # Resize heatmap
    heatmap_image = heatmap_image.resize(
        (
            original_uint8.shape[1],
            original_uint8.shape[0]
        )
    )

    # Convert back to NumPy
    heatmap_array = np.array(
        heatmap_image
    )

    # --------------------------------------------------------
    # Create red heatmap
    # --------------------------------------------------------

    red = heatmap_array

    green = np.zeros_like(
        heatmap_array
    )

    blue = np.zeros_like(
        heatmap_array
    )

    colored_heatmap = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    # --------------------------------------------------------
    # Blend original image + heatmap
    # --------------------------------------------------------

    overlay = (
        original_uint8 * (1 - alpha)
        +
        colored_heatmap * alpha
    )

    # Keep pixels within valid range
    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# 6. SAVE GRAD-CAM
# ============================================================

def save_gradcam(
    image_path,
    model,
    image_size,
    output_path,
    target_class_index=None
):
    """
    Generate and save Grad-CAM visualization.
    """

    # Load image
    original_image, image_tensor = (
        load_and_preprocess_image(
            image_path,
            image_size
        )
    )

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        image_tensor,
        model,
        target_class_index
    )

    # Create overlay
    overlay = create_gradcam_overlay(
        original_image,
        heatmap
    )

    # Create output folder
    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # Save image
    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    print(
        "\nGrad-CAM saved to:"
    )

    print(
        output_path
    )

    return heatmap, overlay