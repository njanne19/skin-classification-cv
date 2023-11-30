import numpy as np 
import torch 
import os 
import torchvision 
import time 
import logging 
import warnings 
from skin_classification_cv.models.vgg import SkinVGG
from skin_classification_cv.loaders.imagenet_loader import ImageNetDataset
from tensorboardX import SummaryWriter
from datetime import datetime

"""
    SkinVGGTrainer is a class that encapsulates the training and evaluation of the SkinVGG model.
    It can be used to train the model, and evaluate it on a validation set, and save weights for future testing. 
"""
class SkinVGGTrainer:
    def __init__(self, model, train_loader, test_loader, val_loader=None, criterion=None, optimizer=None, scheduler=None, device=None):
        """
        SkinVGGTrainer is a class that encapsulates the training and evaluation of the SkinVGG model.
        It can be used to train the model, and evaluate it on a validation set, and save weights for future testing.
        
        Args:
            model (SkinVGG): The SkinVGG model to train and evaluate. 
            train_loader (DataLoader): The DataLoader object that loads the training data. 
            test_loader (DataLoader): The DataLoader object that loads the test data. 
            val_loader (DataLoader): The DataLoader object that loads the validation data. 
            criterion (Loss): The loss function to use. Default: CrossEntropyLoss
            optimizer (Optimizer): The optimizer to use. Default: Adam
            [NOT IMPLEMENTED YET] scheduler (Scheduler): The scheduler to use. Default: StepLR
            device (Device): The device to use. Default: Cuda if available. 
        """
        
        # Now implement this init functionality with the defaults as provided in the doxygen comment above. 
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # If the device is not provided, try to use CUDA if available.
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        # Send model to device 
        self.model.to(self.device)
            
        # If the criterion is not provided, use CrossEntropyLoss.
        if self.criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
            
        # If the optimizer is not provided, use Adam.
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
        # Check to see if there is a weights folder, a runs folder 
        # if not, create them 
        if not os.path.exists('./skin_classification_cv/models/vgg/weights'):
            # Create this directory 
            os.makedirs('./skin_classification_cv/models/vgg/weights')
            
        if not os.path.exists('./skin_classification_cv/models/vgg/runs'):
            # Create this directory 
            os.makedirs('./skin_classification_cv/models/vgg/runs')
            
        # Add logging config 
        logging.basicConfig(level=logging.INFO)
        
        
    def fit(self, epochs=10, show=True, update_step=1, writer=None): 
        """
        fit is a function that encapsulates the training loop for the model. 
        """
        
        # Keep track of the best validation loss 
        best_val_loss = np.inf
        
        # Track total start time
        total_start_time = time.time()
        
        # Iterate over the epochs
        for epoch in range(1, epochs+1): 
            # Start the timer 
            start_time = time.time()
            
            # Train the model 
            tr_stats = self._train(self.train_loader, epoch)
            
            # Validate the model 
            val_loss, val_stats = self._validate(self.val_loader)
            
            # Calculate the epoch time 
            epoch_time = time.time() - start_time
            
            # Log the results 
            self._logger(tr_stats, val_stats, epoch, epochs, epoch_time, show, update_step)
            
            # Add to tensorboard as well if writer is provided 
            if writer is not None:
                writer.add_scalar('Loss/train', tr_stats['loss'], epoch)
                writer.add_scalar('Loss/val', val_stats['loss'], epoch)
                writer.add_scalar('Acc/val', val_stats['acc'], epoch)
            
            # Save the model if the validation loss is the best we've seen so far
            if val_loss < best_val_loss: 
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), './skin_classification_cv/models/vgg/weights/best_model.pt')
                
        
        # Print total time of training
        total_time = time.time() - total_start_time
        logging.info(f"Total training time: {total_time:.2f}s")
                 
        # Return the model back to the caller. 
        return self.model
        
    
    def _train(self, loader, epoch): 
        """
        _train is a private function that encapsulates the training loop for the model.
        """
        
        # Put the model into train mode 
        self.model.train() 
        
        # Keep a running loss total 
        running_loss = 0.0
        
        # Print beignning statistics of this epoch 
        logging.info(f"Beginning epoch {epoch}")
        logging.info(f"Training on {len(loader)} batches")
        
        
        # Iterate over the data loader
        for batch_idx, (features, labels) in enumerate(loader): 
            
            # Move the data to the right device 
            features = features.to(self.device) 
            labels = labels.to(self.device)
            
            # Write a print statement that shows the current batch being trained, the total number of batches
            # and the current loss for the batch. Print in place so as the for loop iterates the loss is updated 
            # in the same spot below. 
            print(f"({epoch}) Training on batch {batch_idx} of {len(loader)} | Current Loss: {running_loss:.4f}", end='\r')
            
            # Forward pass 
            out = self.model(features) 
            
            # calculate the loss 
            loss = self.criterion(out, labels)
            
            # Add the loss to the running loss total
            running_loss += loss.item()
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            self.optimizer.step()
            
        # Create a dictionary of the training statistics
        tr_stats = {
            'loss': running_loss
        }
            
        # Return the loss back to the caller. 
        return tr_stats
    
    def _validate(self, loader): 
        """
        _validate is a private function that encapsulates the validation loop for the model.
        """
        
        # Put the model into eval mode 
        self.model.eval()
        
        # Keep a running loss total
        running_loss = 0.0
        
        # Keep a running accuraccy total as well
        num_correct = 0
        total_samples = 0
        
        with torch.no_grad(): 
            # Iterate over the data loader
            for features, labels in loader: 
                
                # Move the data to the right device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass 
                out = self.model(features) 

                # calculate the loss 
                loss = self.criterion(out, labels)
                
                # Add the loss to the running loss total
                running_loss += loss.item()
                
                # Add num correct and total samples
                _, preds = torch.max(out, 1)
                num_correct += torch.sum(preds == labels.data)
                total_samples += preds.size(0)
            
        # Calculate the accuracy 
        acc = num_correct / total_samples
        
        # Create a dictionary of the validation statistics
        val_stats = {
            'loss': running_loss,
            'acc': acc, 
            'num_correct': num_correct,
            'total_samples': total_samples
        }
                
        # Return the loss back to the caller. 
        return val_stats
    
    
    def _logger(self, tr_stats, val_stats, epoch, epochs, epoch_time, show=True, update_step=20): 
        """
        _logger is a private function that encapsulates the logging functionality for the model.
        """
        
        if show: 
            if epoch % update_step == 0 or epoch == 1: 
                
                # Create the message to log
                msg = f"Epoch: {epoch}/{epochs} | Train Loss: \
                {tr_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f} | \
                | Val Acc: {val_stats['acc']:.2f}% ({val_stats['num_correct']} / {val_stats['total_samples']}) | \
                Epoch Time: {epoch_time:.2f}s"
                
                logging.info(msg)
            
            
# Test to see if training worked 
if __name__ == "__main__": 
        
    # Create the logfile for this specific run 
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Append the datetime string to the logfile 
    log_dir = f'skin_classification_cv/models/vgg/runs/training-{current_time}'
    
    # Initialize the summary writer 
    writer = SummaryWriter(log_dir)
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    
    # First load in HAM10000 and split datasets 
    dataset = ImageNetDataset('datasets/HAM_10000') 
    
    # Split the dataset into train, test, and validation sets
    # into 80%, 10%, and 10% respectively.
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    
    # Create the model based on number of skin classes
    skin_vgg = SkinVGG(dataset.num_classes)
    
    # Load the best model weights if they already exist
    if os.path.exists('./skin_classification_cv/models/vgg/weights/best_model.pt'): 
        print("Found best weights @ ./skin_classification_cv/models/vgg/weights/best_model.pt")
        print("loading weights...")
        skin_vgg.load_state_dict(torch.load('./skin_classification_cv/models/vgg/weights/best_model.pt'))
    
    # Create the data loaders 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
    
    # Create the trainer 
    trainer = SkinVGGTrainer(skin_vgg, train_loader, test_loader, val_loader)
    
    # Train the model 
    trainer.fit(epochs=10, show=True, update_step=1, writer=writer)
    
    # Evaluate the model on the test set
    test_stats = trainer._validate(test_loader)

    
    # Print the results 
    print(f"Test Loss: {test_stats['loss']:.4f} | \
        Test Acc: {test_stats['acc']:.4f} \
        ({test_stats['num_correct']}/{test_stats['total_samples']})")