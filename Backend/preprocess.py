
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define image size
img_size = [256, 256]

# Define paths
image_dir = r"C:\Users\Home\PycharmProjects\pythonProject3\Forest Segmented\images"
mask_dir = r"C:\Users\Home\PycharmProjects\pythonProject3\Forest Segmented\masks"

# Get sorted lists of image and mask file paths
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".jpg")])

# Create DataFrame
df = pd.DataFrame({"image_path": image_paths, "mask_path": mask_paths})


# Function to preprocess images and masks
def preprocess(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.where(mask > 127, 1.0, 0.0)  # Converts masks correctly
    # Binarize mask

    return image, mask


# Data augmentation function
def data_augmentation(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask


# Create dataset function
def create_dataset(df, train=False):
    ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["mask_path"].values))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        ds = ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    return ds


# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=42)  # Now test is separate from validation

# Create datasets
train_dataset = create_dataset(train_df, train=True)
valid_dataset = create_dataset(valid_df)
test_dataset = create_dataset(test_df)

# Training parameters
BATCH_SIZE = 32
BUFFER_SIZE = 1000
TRAIN_LENGTH = len(train_df)

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)


# Visualization function
def display(display_list):
    plt.figure(figsize=(12, 12))
    titles = ['Input Image', 'True Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.show()


if __name__ == "__main__":  # Only run this if preprocess.py is run directly
    print(">>> Running preprocess.py")

    for images, masks in train_dataset.take(1):
        plt.imshow(images[0])
        plt.show()

    for image, mask in train_dataset.take(1):
        display([image[0], mask[0]])

    print(">>> preprocess.py finished.")
    