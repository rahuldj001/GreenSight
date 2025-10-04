import tensorflow as tf
from tensorflow.keras import layers, Model

# ✅ Define a Named Function Instead of Lambda
def cast_to_float32(img):
    return tf.cast(img, tf.float32)

def unet_model(output_channels):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

    layer_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
        "block_16_project",
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    inputs = tf.keras.Input(shape=[256, 256, 3])

    # ✅ Use Named Function Instead of Lambda
    x = tf.keras.layers.Lambda(cast_to_float32, name="cast_to_float32")(inputs)

    skips = down_stack(x)
    x = skips[-1]

    def upsample(filters, size):
        return tf.keras.Sequential([
            layers.Conv2DTranspose(filters, size, strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
        ])

    up_stack = [
        upsample(320, 3),
        upsample(192, 3),
        upsample(144, 3),
        upsample(96, 3),
    ]

    for up, skip in zip(up_stack, reversed(skips[:-1])):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    last = layers.Conv2DTranspose(output_channels, 3, strides=2, padding="same", activation="sigmoid")
    outputs = last(x)

    return Model(inputs=inputs, outputs=outputs)

# ✅ Save the Model in `.keras` Format
model = unet_model(output_channels=1)
model.save("unet_deforestation_model.keras", save_format="keras")
print("✅ Model saved in .keras format successfully!")
