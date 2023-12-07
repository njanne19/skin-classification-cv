import numpy as np 
import torch 
import torchvision 
from torchsummary import summary

class SkinVGG(torch.nn.Module):
    def __init__(self, num_classes=2):
        
        # Call the parent init function
        super(SkinVGG, self).__init__()
        
        # Load the pre-trained part of the model. 
        self.vgg = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
        
        # Now we need to adapt the model to fit our needs. We are going 
        # to replace the last fully connected layer with an FC layer that matches
        # the number of classes in our dataset. Then we are going to freeze all 
        # the layers of VGG before this layer, so that we only train the last layer. 
        
        # Begin by freezing the pre-trained section all the but the 
        # last 2 convolutional layers. 
        for param in self.vgg.features[:-2].parameters():
            param.requires_grad = False
        
        # Get the number of input features for the last layer
        num_features = self.vgg.classifier[6].in_features
        
        # Replace the last layer with a new fully connected layer that has
        # num_classes outputs
        self.vgg.classifier[6] = torch.nn.Linear(num_features, num_classes)

        # Add log softmax for prediction
        self.vgg.classifier.add_module('7', torch.nn.LogSoftmax(dim=1)) 
        

    def forward(self, x):
        return self.vgg(x)
    
    
if __name__ == "__main__": 
    
    device = torch.cuda.current_device()
    
    skin_vgg = SkinVGG(6)
    skin_vgg.to(device) 

    print('The model:')
    print(skin_vgg)

    print('\n\nModel params:')
    summary(skin_vgg, (3, 224, 224))