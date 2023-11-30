import numpy as np
import  pandas as pd
import os

from torchvision import transforms
from PIL import Image

class EigenfacesDataset():
    def __init__(self, dir, metadata_file='HAM10000_metadata.csv', transform=None):
        
        self.dir = dir
        
        metadata_filepath = os.path.join(dir, metadata_file)
        try: 
            self.metadata = pd.read_csv(metadata_filepath)
        except: 
            raise Exception(f'Metadata filepath ({metadata_filepath}) not found. Please check that the raw data directory is correct.')
        
        self.transform = transform

        if self.transform is None:
            
            # Uses the same transformer as the ImageNet/VGG model
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def collect_images(self): 

        ids = self.metadata['image_id']
        dx = np.unique(self.metadata['dx']) # get diagnosis for record
        code = dict(zip(dx, range(len(dx)))) # create interger encoding of diagnosis type

        y = np.zeros(len(ids)) # initialize X and y
        X = np.zeros((len(ids), 150528))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        transform = self.transform
        
        counter = 0
        for i in ids:
            if int(i[-5:]) < 29306: # get image from correct folder
                img_name = f'{self.dir}/HAM10000_images_part_1/{i}.jpg'
            else:
                img_name = f'{self.dir}/HAM10000_images_part_2/{i}.jpg'
            
            image = Image.open(img_name).convert('RGB')
            image = transform(image) # crop image and scale down
            inp = image.numpy().transpose((1, 2, 0))
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

            vec = inp.reshape(-1)
            X[counter] = vec 
            y[counter] = code[self.metadata[self.metadata['image_id'] == i]['dx'].values[0]]
            counter += 1

        return X, y

