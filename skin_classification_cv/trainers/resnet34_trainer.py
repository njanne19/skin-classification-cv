import numpy as np 
import torch 
import os 
import torchvision 
import pdb
import time 
import logging 
import warnings 
from skin_classification_cv.models.resnet34 import SkinResNet34
from skin_classification_cv.loaders.imagenet_loader import ImageNetDataset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from datetime import datetime

"""
    SkinResNet34Trainer is a class that encapsulates the training and evaluation of the SkinResNet34 model.
    It can be used to train the model, and evaluate it on a validation set, and save weights for future testing. 
"""
class SkinResNet34Trainer:
    def __init__(self, model, train_loader, test_loader, val_loader=None, criterion=None, optimizer=None, scheduler=None, device=None):
        """
        SkinResNet34Trainer is a class that encapsulates the training and evaluation of the SkinResNet34 model.
        It can be used to train the model, and evaluate it on a validation set, and save weights for future testing.
        
        Args:
            model (SkinResNet34): The SkinResNet34 model to train and evaluate. 
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
            self.criterion = torch.nn.NLLLoss()
            
        # If the optimizer is not provided, use Adam.
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
        # Check to see if there is a weights folder, a runs folder 
        # if not, create them 
        if not os.path.exists('./skin_classification_cv/models/resnet34/weights'):
            # Create this directory 
            os.makedirs('./skin_classification_cv/models/resnet34/weights')
            
        if not os.path.exists('./skin_classification_cv/models/resnet34/runs'):
            # Create this directory 
            os.makedirs('./skin_classification_cv/models/resnet34/runs')
            
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
            val_stats = self._validate(self.val_loader)
            
            # Calculate the epoch time 
            epoch_time = time.time() - start_time
            
            # Log the results 
            self._logger(tr_stats, val_stats, epoch, epochs, epoch_time, show, update_step)
            
            # Add to tensorboard as well if writer is provided 
            if writer is not None:
                writer.add_scalar('Loss/train', tr_stats['loss'], epoch)
                writer.add_scalar('Loss/val', val_stats['loss'], epoch)
                writer.add_scalar('Acc/val', val_stats['acc'], epoch)
                writer.flush()
            
            # Save the model if the validation loss is the best we've seen so far
            if val_stats['loss'] < best_val_loss: 
                best_val_loss = val_stats['loss']
                torch.save(self.model.state_dict(), './skin_classification_cv/models/resnet34/weights/best_model.pt')
                
        
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
        losses = []
        
        # Print beignning statistics of this epoch 
        logging.info(f"Beginning epoch {epoch}")
        logging.info(f"Training on {len(loader)} batches")
        
        
        # Iterate over the data loader
        for batch_idx, (features, labels) in enumerate(loader): 
            
            # Move the data to the right device 
            features = features.to(self.device) 
            labels = labels.to(self.device)
        
            
            # Zero the gradients, apparently this way is better than zero_grad()
            for param in self.model.parameters():
                param.grad = None

            # Forward pass 
            out = self.model(features) 
            
            # calculate the loss 
            loss = self.criterion(out, labels)
            
            # Add the loss to the running loss total
            losses.append(loss.item())

            # breakpoint()
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            self.optimizer.step()

            # Write a print statement that shows the current batch being trained, the total number of batches
            # and the current loss for the batch. Print in place so as the for loop iterates the loss is updated 
            # in the same spot below. 
            print(f"({epoch}) Training on batch {batch_idx} of {len(loader)} | Current batch loss: {loss.item():.4f}", end='\r')

            
        # Create a dictionary of the training statistics
        tr_stats = {
            'loss': np.mean(losses)
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
        losses = []
        
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
                losses.append(loss.item())   
                
                # Note the number correct
                predictions = torch.argmax(out, 1)
                num_correct += torch.sum(predictions == labels)
                total_samples += len(labels)
            
        # Calculate the accuracy 
        acc = (num_correct / total_samples) * 100.0
        
        # Create a dictionary of the validation statistics
        val_stats = {
            'loss': np.mean(losses),
            'acc': acc, 
            'num_correct': num_correct,
            'total_samples': total_samples
        }
                
        # Return the loss back to the caller. 
        return val_stats
    
    def k_fold_train(self, dataset, k=5, epochs=10, show=True, update_step=1, writer=None):
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_performance = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            # Define data loaders for training and validation phases
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler, num_workers=6, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_subsampler, num_workers=6, pin_memory=True)

            # Set data loaders
            self.train_loader = train_loader
            self.val_loader = val_loader

            # Fit model
            model = self.fit(epochs, show=show, update_step=update_step, writer=writer)

            # Evaluate and store performance
            val_stats = self._validate(self.val_loader)
            fold_performance.append(val_stats['acc'])

            print(f"Fold {fold} Accuracy: {val_stats['acc']}%")
        
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for idx, accuracy in enumerate(fold_performance):
            print(f'Fold {idx}: {accuracy} %')
            sum += accuracy
        print(f'Average: {sum/len(fold_performance)} %')
    
    
    def _logger(self, tr_stats, val_stats, epoch, epochs, epoch_time, show=True, update_step=20): 
        """
        _logger is a private function that encapsulates the logging functionality for the model.
        """
        
        if show: 
            if epoch % update_step == 0 or epoch == 1: 
                
                # Create the message to log
                msg = f"Epoch: {epoch}/{epochs} | Train Loss (per batch): {tr_stats['loss']:.4f} | Val Loss (per batch): {val_stats['loss']:.4f} |"
                msg += f"| Val Acc: {val_stats['acc']:.2f}% ({val_stats['num_correct']} / {val_stats['total_samples']}) |"
                msg += f"Epoch Time: {epoch_time:.2f}s"
                
                logging.info(msg)
            
            
# Test to see if training worked 
if __name__ == "__main__": 
        
    # Create the logfile for this specific run 
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Append the datetime string to the logfile 
    log_dir = f'skin_classification_cv/models/resnet34/runs/training-{current_time}'
    
    # Initialize the summary writer 
    writer = SummaryWriter(log_dir)
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    
    # First load in HAM10000 and split datasets 
    dataset = ImageNetDataset('datasets/HAM_10000', augment_factor=int(2)) 
    
    # Split the dataset into train, test, and validation sets
    # into 80%, 10%, and 10% respectively.
    train_set, test_set = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=generator)
    
    # Create the model based on number of skin classes
    skin_resnet34 = SkinResNet34(dataset.num_classes)
    
    # Load the best model weights if they already exist
    if os.path.exists('./skin_classification_cv/models/resnet34/weights/best_model.pt'): 
        print("Found best weights @ ./skin_classification_cv/models/resnet34/weights/best_model.pt")
        print("loading weights...")
        skin_resnet34.load_state_dict(torch.load('./skin_classification_cv/models/resnet34/weights/best_model.pt'))
    
    # Create the data loaders 
    # Options after shuffle may only work on GPU. If this is causing errors, remove them. 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    
    # Create the trainer 
    trainer = SkinResNet34Trainer(skin_resnet34, None, test_loader, None)
    
    # Train the model 
    trainer.k_fold_train(train_set, epochs=100, show=True, update_step=1, writer=writer)

    # Load in the best model
    skin_resnet34.load_state_dict(torch.load('./skin_classification_cv/models/resnet34/weights/best_model.pt'))
    
    # Evaluate the model on the test set
    test_stats = trainer._validate(test_loader)

    
    # Print the results 
    print(f"Test Loss: {test_stats['loss']:.4f} | \
        Test Acc: {test_stats['acc']:.4f} \
        ({test_stats['num_correct']}/{test_stats['total_samples']})")
