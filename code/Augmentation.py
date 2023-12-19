from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import os

# Set the path to your dataset
train_data_dir = '/Users/shahadsaeed/Desktop/melanoma_cancer_dataset/train'

# Set the path to store augmented images on the desktop
augmented_data_dir = '/Users/shahadsaeed/Desktop/augmented_data'

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to perform data augmentation for a specific class
def augment_data(class_folder, class_name):
    class_path = os.path.join(train_data_dir, class_folder)

    # Make sure the augmented data directory exists
    augmented_class_dir = os.path.join(augmented_data_dir, class_name)
    os.makedirs(augmented_class_dir, exist_ok=True)

    # Get a list of all images in the class folder
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(".jpg")]

    # Iterate through the images and generate augmented images
    for img_path in images:
        # Load the image
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images and save them
        for i in range(1):  # Set the desired number of augmentations for each image
            augmented_img = datagen.random_transform(x[0])
            augmented_img_path = os.path.join(augmented_class_dir, f'aug_{i}_{os.path.basename(img_path)}')
            array_to_img(augmented_img).save(augmented_img_path)

# Augment the malignant class in the training dataset
augment_data('malignant', 'malignant_augmented')

# Augment the benign class in the training dataset
augment_data('benign', 'benign_augmented')
