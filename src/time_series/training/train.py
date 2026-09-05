# import dependencies
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from src.time_series.models.conv_autoencoder import ConvAutoencoder, masked_mse



def train_model(x_train: np.ndarray, m_train: np.ndarray, x_val:np.ndarray, m_val: np.ndarray, epochs: int= 150,
                batch_size: int= 32, lr: float= 1e-3) -> tuple[ConvAutoencoder, list[float]]:
    
    # convert the numpy arrays into tensor also build tensordata and dataloader for trainig data
    x_train_t= torch.tensor(x_train, dtype=torch.float32)
    m_train_t= torch.tensor(m_train, dtype=torch.float32)
    x_val_t= torch.tensor(x_val, dtype= torch.float32)
    m_val_t= torch.tensor(m_val, dtype= torch.float32)
    
    # build tensor data
    train_ds= TensorDataset(x_train_t, m_train_t)
    train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle= True)
    # intialize the model and an adam optimizer.
    model= ConvAutoencoder()
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    
    # history of the loss so we can plot it later on as well
    training_losses: list[float]= []
    validation_losses: list[float]= []
    
    # run the model in epochs
    for epoch in range(epochs):
        # make model in train mode.
        model.train()
        # initalize the running loss
        running_loss= 0.0
        # iterate over the data
        for xb, mb in train_loader:
            # set the gradients to zero
            optimizer.zero_grad()
            # reconstruct
            recon= model(xb, mb)
            loss= masked_mse(recon= recon, x=xb, mask= mb)
            # backpropogation
            loss.backward()
            # step in optimizer
            optimizer.step()
            # add the batch loss in total_loss
            running_loss += loss.item()
        # calculate what it avg loss per batch
        avg_train_loss= running_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # validation phase (no gradient updates)
        model.eval()
        with torch.no_grad():
            val_recon= model(x_val_t, m_val_t)
            val_loss= masked_mse(val_recon, x_val_t, m_val_t)
        validation_losses.append(val_loss.item())
        
    return model, training_losses, validation_losses 
            