import numpy as np 
import torch 
import torchvision 
from torchsummary import summary

class SkinResNet50(torch.nn.Module):
    def __init__(self, num_classes=2):
        
        # Call the parent init function
        super(SkinResNet50, self).__init__()
        
        # Load the pre-trained part of the model. 
        self.resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Now we need to adapt the model to fit our needs. We are going 
        # to replace the last fully connected layer with an FC layer that matches
        # the number of classes in our dataset. Then we are going to freeze all 
        # the layers of VGG before this layer, so that we only train the last layer. 
        
        # Begin by freezing the pre-trained section all the but the 
        # last 2 convolutional layers. 
        # for param in self.resnet.features[:-2].parameters():
        #     param.requires_grad = False
        
        # Get the number of input features for the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the last layer with a new fully connected layer that has
        # num_classes outputs
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)

        # Add log softmax for prediction
        self.softmax = torch.nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        pre_softmax = self.resnet(x) 
        return self.softmax(pre_softmax)
    
    
if __name__ == "__main__": 
    
    device = torch.cuda.current_device()
    
    skin_vgg = SkinResNet50(7)
    skin_vgg.to(device) 

    print('The model:')
    print(skin_vgg)

    print('\n\nModel params:')
    summary(skin_vgg, (3, 224, 224))