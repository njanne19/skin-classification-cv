import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

class ImageNetDataset(Dataset):
    def __init__(self, dir, metadata_file='HAM10000_metadata.csv', transform=None):
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
            
        # Get number of unique classes to vectorize labels
        self.classes = self.data_frame.iloc[:, 2].unique()
        self.num_classes = len(self.classes)
        
    def class_label_to_class_vector(self, label):
        # Find the index of the label in the classes array 
        idx = np.where(self.classes == label)[0][0]
        
        # Create a vector of zeros of length num_classes
        vec = np.zeros(self.num_classes)
        
        # Set the index of the label to 1
        vec[idx] = 1
        
        # Return vector to user
        return vec
    
    def class_vector_to_class_label(self, vec):
        # Pic the idx of the max value in the vector
        idx = np.argmax(vec)
        
        # Return the class label at that index
        return self.classes[idx]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, (self.data_frame.iloc[idx, 1] + '.jpg'))
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, self.class_label_to_class_vector(label)


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
    dataset = ImageNetDataset('datasets/HAM_10000')

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
        imshow(img, title=str(dataset.class_vector_to_class_label(label)))
        
    # Hold the window open 
    plt.show()
    
    # Print cwd 
    print(os.getcwd())

    
