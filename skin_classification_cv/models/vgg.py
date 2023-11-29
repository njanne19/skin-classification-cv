import numpy as np 
import torch 
import torchvision 
from torchsummary import summary

class SkinVGG(torch.nn.Module):
    def __init__(self, num_classes=2):
        
        # Call the parent init function
        super(SkinVGG, self).__init__()
        
        # Load the pre-trained part of the model. 
        self.vgg = torchvision.models.vgg16()
        
        # Now we need to adapt the model to fit our needs. We are going 
        # to replace the last fully connected layer with an FC layer that matches
        # the number of classes in our dataset. Then we are going to freeze all 
        # the layers of VGG before this layer, so that we only train the last layer. 
        
        # Begin by freezing the pre-trained section 
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Get the number of input features for the last layer
        num_features = self.vgg.classifier[-1].in_features
        
        # Replace the last layer with a new fully connected layer that has
        # num_classes outputs
        self.vgg.classifier[-1] = torch.nn.Linear(num_features, num_classes)
        

    def forward(self, x):
        return self.vgg(x) 
    
    
if __name__ == "__main__": 
    
    skin_vgg = SkinVGG(6)

    print('The model:')
    print(skin_vgg)

    print('\n\nModel params:')
    summary(skin_vgg, (3, 224, 224))