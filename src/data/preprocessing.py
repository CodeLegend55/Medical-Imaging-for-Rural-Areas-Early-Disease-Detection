from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32):
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x / 255.0,  # Normalize to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=lambda x: x / 255.0)
    test_datagen = ImageDataGenerator(preprocessing_function=lambda x: x / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image = preprocess_image(img_path)
                images.append(image)
                labels.append(class_names.index(class_name))

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names