import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createDataset(num=1024):
    # create a dataset
    X = np.random.uniform(0, 2 * np.pi, num)
    Y = np.sin(X) + np.random.normal(0, 0, num)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)

    # shuffle the dataset 
    indices = torch.randperm(num)
    X = X[indices]
    Y = Y[indices]

    return X, Y

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                num_layers=2, 
                activation=nn.Sigmoid()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        if num_layers < 2:
            raise ValueError("Number of layers must be greater than 1")

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.activation = activation
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def test_model(model, X_val, Y_val, criterion=nn.MSELoss()):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, Y_val)
    return val_loss.item()

def train(X, Y, hyper_param):
    num_data = X.shape[0]

    X_train, X_val = torch.split(X, [int(0.9 * num_data), num_data - int(0.9 * num_data)], dim=0)
    Y_train, Y_val = torch.split(Y, [int(0.9 * num_data), num_data - int(0.9 * num_data)], dim=0)

    # unpack hyper parameters
    num_epochs = hyper_param["num_epochs"]
    lr = hyper_param["lr"]
    batch_size = hyper_param["batch_size"]
    
    model = MLP(hyper_param['input_size'], 
                hyper_param['hidden_size'], 
                hyper_param['output_size'],
                hyper_param['num_layers'],
                hyper_param['activation']
                ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_param['lr'])

    # create a dataloader
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create scheduler
     # Learning rate scheduler
    if "gamma" in hyper_param:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyper_param['gamma'])
    else:
        scheduler = None

    # train the model
    train_losses = []
    val_losses = []
    # Progress bar for the number of epochs
    progress_bar = tqdm(range(hyper_param['num_epochs']), desc="Training progress")
    
    for epoch in progress_bar:
        model.train()  # Set model to training mode

        # Training loop
        for X_batch, Y_batch in dataloader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Record the last batch's training loss
        train_losses.append(loss.item())
        
        # Validation step after each epoch
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            val_losses.append(val_loss.item())
        
        # Update learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Update progress bar description every epoch with train and validation loss
        if(epoch % 10 == 0):
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.3e}", "Val Loss": f"{val_loss.item():.3e}"})
            progress_bar.update(10)
         
    progress_bar.close()
    return train_losses, val_losses, model

def plot_loss(train_losses, val_losses):
    def smooth_curve(values, smoothing_factor=0.99):
        smoothed_values = []
        last = values[0]
        for value in values:
            smoothed_value = last * smoothing_factor + (1 - smoothing_factor) * value
            smoothed_values.append(smoothed_value)
            last = smoothed_value
        return smoothed_values
    
    train_losses_smoothed = smooth_curve(train_losses)
    val_losses_smoothed = smooth_curve(val_losses)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_smoothed, label="Training Loss (Smoothed)", color="blue")
    plt.plot(val_losses_smoothed, label="Validation Loss (Smoothed)", color="red")
    plt.yscale("log")  # Log scale for the y-axis
    plt.xlabel("Epoch")
    plt.ylabel("MSE (Log Scale)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


if __name__ == "__main__":
    num = 1024
    X, Y = createDataset(num)

    X_train, X_test = torch.split(X, [int(0.8 * num), num - int(0.8 * num)], dim=0)
    Y_train, Y_test = torch.split(Y, [int(0.8 * num), num - int(0.8 * num)], dim=0)

    init_lr = 1e-2
    final_lr = 1e-6
    epochs = 15000
    gamma = (final_lr / init_lr) ** (1 / epochs)

    hyper_param = {
        "input_size": 1,
        "hidden_size": 64,
        "output_size": 1,
        "num_layers": 4,
        "activation": nn.Sigmoid(),
        "num_epochs": epochs,
        "lr": init_lr,
        "batch_size": 128,
        "gamma": gamma
    }

    print(hyper_param)
    train_losses, val_losses, model = train(X_train, Y_train, hyper_param)
    plot_loss(train_losses, val_losses)

    test_loss = test_model(model, X_test, Y_test)
    print(f"Test Loss: {test_loss:.3e}")


