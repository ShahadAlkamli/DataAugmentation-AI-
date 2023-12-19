import os
import shutil

# Set the path to your dataset
base_path = '/Users/shahadsaeed/Desktop/melanoma_cancer_dataset/'
train_path = os.path.join(base_path, 'train')
augmented_train_path = os.path.join(base_path, 'augmentation')

# Define the merged folder
merged_train_path = os.path.join(base_path, 'augmenetd_train')

# Create the merged folder if it doesn't exist
if not os.path.exists(merged_train_path):
    os.makedirs(merged_train_path)

# Function to merge the contents of source folder into destination folder
def merge_folders(src_folder, dest_folder):
    for class_name in os.listdir(src_folder):
        src_class_path = os.path.join(src_folder, class_name)
        dest_class_path = os.path.join(dest_folder, class_name)

        # Skip hidden files and directories
        if class_name.startswith('.'):
            continue

        # Create the destination class folder if it doesn't exist
        if not os.path.exists(dest_class_path):
            os.makedirs(dest_class_path)

        # Print the list of files before copying
        files_to_copy = [filename for filename in os.listdir(src_class_path) if not filename.startswith('.')]
        print(f"Copying {len(files_to_copy)} files from {src_class_path} to {dest_class_path}")

        # Copy all files from the source class folder to the destination class folder
        for filename in files_to_copy:
            src_file_path = os.path.join(src_class_path, filename)
            dest_file_path = os.path.join(dest_class_path, filename)
            shutil.copy(src_file_path, dest_file_path)


# Merge the train and augmented_train folders
merge_folders(train_path, merged_train_path)
merge_folders(augmented_train_path, merged_train_path)

# Verify that the merge is successful
merged_malignant_path = os.path.join(merged_train_path, 'malignant')
merged_benign_path = os.path.join(merged_train_path, 'benign')

print(f"Number of merged malignant images: {len(os.listdir(merged_malignant_path))}")
print(f"Number of merged benign images: {len(os.listdir(merged_benign_path))}")
