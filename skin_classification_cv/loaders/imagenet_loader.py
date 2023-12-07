import os
import pdb
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import fiftyone as fo
import cv2

class ImageNetDataset(Dataset):
    def __init__(self, dir, metadata_file='HAM10000_metadata.csv', filename_column=1, label_column=2, transform=None, augment_factor=int(1)):
        """
        ImageNetDataset is an adapter class that loads images from 
        the downloaded dataset and adapts them to the standard ImageNet format 
        to enable transfer learning with off-the-shelf models. The ImageNet format is 
        a (224, 224, 3) image with RGB channels and normalized with the ImageNet mean and std.
        
        Args:
            dir (string): Directory with all the images and metadata file.
            metadata_file (string): Name of the metadata file. Default: 'HAM10000_metadata.csv', for the HAM10000
            transform (callable, optional): Optional transform to be applied on an image.
        """
        
        # Get metadata file path and try to see if 
        # we can load from it. 
        metadata_filepath = os.path.join(dir, metadata_file)
        
        try: 
            self.data_frame = pd.read_csv(metadata_filepath)
        except: 
            raise Exception(f'Metadata filepath ({metadata_filepath}) not found. Please check that the raw data directory is correct.')
       
        self.img_dir = dir
        
        # Get the column indices for the filename and label columns
        self.filename_column = filename_column
        self.label_column = label_column
        
        # Check to see if we need to append file extension 
        file_extensions = ['.jpg', '.png', '.jpeg']
        
        # If the filename does not have an extension, set a flag 
        self.file_extension = '.jpg'
        for extension in file_extensions: 
            if self.data_frame.iloc[0, self.filename_column].endswith(extension): 
                self.file_extension = ''
                break
        
        # Set up transform 
        self.transform = transform

        if self.transform is None:
            
            # Do the transformation here. 
            # Note that these means/std are for the ImageNet dataset, 
            # and should be hardcoded in, as seen below. 
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # Create augmented transform to do data augmentation to increase the ratio of the dataset 
        self.augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])            
        ])

        # This is the factor we multiply by the length of the original dataset 
        # To get an augmented dataset. At the end, the resulting dataset will be 
        # of length len(original_dataset) * augment_factor
        self.augment_factor = augment_factor
            
        # Get number of unique classes to vectorize labels
        self.classes = self.data_frame.iloc[:, self.label_column].unique()
        self.num_classes = len(self.classes)
        
        # Then construct fiftyone dataset object for visualization
        self.construct_fiftyone_dataset() 
        
    def class_label_to_class_number(self, label):
        # Map the class label to the index number
        return np.argmax(self.classes == label) 
    
    def class_number_to_class_label(self, vec):
        # Return the class label at that index
        return self.classes[idx]

    def get_class_mapping(self): 
        # Return a dictionary mapping class labels to class vectors
        return {class_label: self.class_label_to_class_number(class_label) for class_label in self.classes}

    def __len__(self):
        return len(self.data_frame) * self.augment_factor

    def __getitem__(self, idx):

        # Check to see if we're doing augmentation or not
        # This is normal mode
        if self.augment_factor == 1:
            img_name = os.path.join(self.img_dir, (self.data_frame.iloc[idx, self.filename_column] + self.file_extension))
            image = Image.open(img_name).convert('RGB')
            label = self.data_frame.iloc[idx, self.label_column]

            if self.transform:
                image = self.transform(image)

            return image, self.class_label_to_class_number(label)
        
        else: 
            # Calculate the original index of the image 
            original_idx = idx % len(self.data_frame) 

            # Get the image name and label
            img_name = os.path.join(self.img_dir, (self.data_frame.iloc[original_idx, self.filename_column] + self.file_extension))
            image = Image.open(img_name).convert('RGB')
            label = self.data_frame.iloc[original_idx, self.label_column]

            # Apply the augmentation transform to images after the original index length
            if idx >= len(self.data_frame):
                image = self.augment_transform(image)
            else: 
                image = self.transform(image)

        return image, self.class_label_to_class_number(label)
    

    def construct_fiftyone_dataset(self): 
        # Construct a fiftyone dataset object for visualization
        # This does not include augmented images. 
    
        # Create a dataset object 
        self.fo_dataset = fo.Dataset()
        
        # Then iterate through every image
        for idx in range(len(self.data_frame)): 
            
            # Get the image name and label
            img_name = os.path.join(self.img_dir, (self.data_frame.iloc[idx, self.filename_column] + self.file_extension))
            label = self.data_frame.iloc[idx, self.label_column]
            
            # Add the sample to the dataset
            new_sample = fo.Sample(filepath=img_name)
            
            # Then add label and metadata information 
            new_sample['diagnosis'] = fo.Classification(label=label)
            new_sample['skin_tone_estimate'] = estimate_skin_tone(img_name)
            
            self.fo_dataset.add_sample(new_sample)
    
        # At the end, compute metadata
        self.fo_dataset.compute_metadata()
        
def estimate_skin_tone(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Define the range for skin color in YCbCr
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)

    # Find skin region in the image
    skin_region = cv2.inRange(ycbcr_image, min_YCrCb, max_YCrCb)

    # Extract the Cr channel
    Cr_channel = ycbcr_image[:,:,2]

    # Calculate the average value of the Cr channel in the skin region
    average_Cr = np.mean(Cr_channel[skin_region > 0])

    return average_Cr if not np.isnan(average_Cr) else 0


if __name__ == "__main__": 
    
    # Example usage below 
    
    import matplotlib.pyplot as plt
    import torchvision
    import random 

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Create the dataset instance
    # dataset = ImageNetDataset('datasets/HAM_10000')
    dataset = ImageNetDataset(
        dir='datasets/ddidiversedermatologyimages',
        metadata_file='ddi_metadata_clean.csv', 
        filename_column=1, 
        label_column=5)

    # Indexing an element
    img, label = dataset[0]  # Get the first image and label

    print(dataset.classes)
    print(label)

    # Displaying the image
    imshow(img, title=str(label))

    # Displaying multiple images
    fig = plt.figure(figsize=(25, 4))
    
    # Select 5 random indices
    random_indices = random.sample(range(len(dataset)), 5)

    # Display images at these indices
    for idx, dataset_idx in enumerate(random_indices):
        img, label = dataset[dataset_idx]
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        imshow(img, title=f"{dataset.class_number_to_class_label(label)}, {label}")
        
        
    # Displaying the fiftyone dataset
    tones = dataset.fo_dataset.sort_by("skin_tone_estimate", reverse=False)
    session = fo.launch_app(view=tones)
        
    # Hold the window open 
    plt.show()
    
    # Print cwd 
    print(os.getcwd())

    
