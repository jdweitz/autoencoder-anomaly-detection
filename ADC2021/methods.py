import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.utils.prune as prune
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load data
class SignalDataset(Dataset):
    def __init__(self, signal_files, signal_labels): # signals is a list of all the signal datasets you want to include
        signal_data = []
        signal_scaled_data = []  # new list for scaled data
        for i, label in enumerate(signal_labels):
            with h5py.File(signal_files[i], 'r') as file:
                test_data = np.array(file['data'])
                test_scaled_data = np.array(file['scaled_data'])  # reading the scaled data
                
            signal_data.append(test_data)
            signal_scaled_data.append(test_scaled_data)  # appending the scaled data

        self.data = np.squeeze(np.array(signal_data)) # save signal data as np array
        self.scaled_data = np.squeeze(np.array(signal_scaled_data))  # save scaled signal data as np array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = torch.from_numpy(self.data[index]).type(torch.float32)
        scaled_data_sample = torch.from_numpy(self.scaled_data[index]).type(torch.float32)  # get scaled data
        return data_sample, scaled_data_sample  # return both the original data and its scaled version

class BkgDataset(Dataset):
    def __init__(self, bkg_file, split):
        with h5py.File(bkg_file, 'r') as file:
            data = np.array(file['data'])
            data_target = np.array(file['data_target'])

            total_length = len(data)
            train_boundary = int(0.4 * total_length)
            val_boundary = int(0.5 * total_length)

            # Adjusting for the 40-10-50 split:
            if split == 'train':
                self.data = data[:train_boundary]
                self.data_target = data_target[:train_boundary]
            elif split == 'val':
                self.data = data[train_boundary:val_boundary]
                self.data_target = data_target[train_boundary:val_boundary]
            else:  # 'test'
                self.data = data[val_boundary:]
                self.data_target = data_target[val_boundary:]

            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data_target = torch.tensor(self.data_target, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return both data and data_target
        return self.data[index], self.data_target[index]

def get_dataloaders(path, BATCH_SIZE = 1024):
    bkg_file = path + "all_background_processed.h5"
    signal_1 = path + 'all_leptoquark_processed.h5'
    signal_2 = path + 'all_Ato4l_processed.h5'
    signal_3 = path + 'all_hChToTauNu_processed.h5'
    signal_4 = path + 'all_hToTauTau_processed.h5'

    # add correct signal labels
    signal_labels = ['leptoquark','Ato4l','hChToTauNu', 'hToTauTau']

    # add correct path to signal files
    signal_files = [signal_1, signal_2, signal_3, signal_4]

    # create datasets
    list_of_signal_dataloaders = []
    for i in range(4):
      #init datasets
      dataset = SignalDataset([signal_files[i]],[signal_labels[i]])
      #make loader
      dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=True) # different than actual batch size
      #add loader to list
      list_of_signal_dataloaders.append(dataloader)

    # Initialize background data
    bkg_trainloader = DataLoader(BkgDataset(bkg_file,'train'), batch_size=BATCH_SIZE, shuffle=True)
    bkg_valloader = DataLoader(BkgDataset(bkg_file,'val'), batch_size=BATCH_SIZE, shuffle=False)
    bkg_testloader = DataLoader(BkgDataset(bkg_file,'test'), batch_size=BATCH_SIZE, shuffle=False)

    return list_of_signal_dataloaders, bkg_trainloader, bkg_valloader, bkg_testloader, signal_labels

# Model, train, test
class Autoencoder(nn.Module):
    def __init__(self, input_size = 57, latent_dim=3):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            self._init_weights(nn.Linear(input_size, 32)),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            self._init_weights(nn.Linear(32, 16)),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            self._init_weights(nn.Linear(16, latent_dim))
        )
        # Decoder
        self.decoder = nn.Sequential(
            self._init_weights(nn.Linear(latent_dim, 16)),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            self._init_weights(nn.Linear(16, 32)),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            self._init_weights(nn.Linear(32, input_size))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _init_weights(self, layer): # implemented this to mimic the HeUniform in tf model
        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer


def train(model, train_loader, num_epochs, lr = 0.00001, early_stopping_patience=10, get_losses = False, val_loader = None): # changed lr to .00001 from .0001, as done in ref
    # Init training optimizers
    optimizer = optim.Adam(model.parameters(), lr = lr)
    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True, min_lr=1E-6)
    #criterion = nn.MSELoss()
    criterion = custom_MSE
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    device = next(model.parameters()).device # added for cuda

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, data_scaled in train_loader:
            data = data.float().to(device)  # ensure data is on the same device as the model (added for cuda)
            data_scaled = data_scaled.float().to(device) # target
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data_scaled) # data_scaled is target
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('epoch [{}/{}], Training loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_loader)))
        if get_losses == True:
            val_loss = test(model, val_loader)
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)

            # Call the scheduler step with the current validation loss
            scheduler.step(val_loss)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered after {} epochs".format(epoch + 1))
                break

    if get_losses == True:
        return train_losses, val_losses

def test(model, dataloader):
  # Testing the autoencoder
  model.eval()
  #criterion = nn.MSELoss()
  criterion = custom_MSE
  test_loss = 0
  with torch.no_grad():
      for data, data_scaled in dataloader:
          data = data.float().to(device)  # Move data to the device that the model is on
          data_scaled = data_scaled.float().to(device)
          outputs = model(data)
          loss = criterion(outputs, data_scaled)
          test_loss += loss.item()

  test_loss = test_loss / len(dataloader)

  print('Loss: {:.4f}'.format(test_loss))
  return test_loss

# Loss, FPR, TPR
def custom_MSE(inputs, outputs):
    inputs = inputs.reshape((inputs.shape[0], 19, 3, 1))
    outputs = outputs.reshape((outputs.shape[0], 19, 3, 1))

    # trick on phi
    outputs_phi = torch.pi * torch.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0 * torch.tanh(outputs)
    outputs_eta_muons = 2.1 * torch.tanh(outputs)
    outputs_eta_jets = 4.0 * torch.tanh(outputs)
    outputs_eta = torch.cat([outputs[:,0:1,:], outputs_eta_egamma[:,1:5,:], outputs_eta_muons[:,5:9,:], outputs_eta_jets[:,9:19,:]], dim=1)
    outputs = torch.cat([outputs[:,:,0], outputs_eta[:,:,1], outputs_phi[:,:,2]], dim=2)

    # change input shape
    inputs = torch.squeeze(inputs, -1)

    # calculate and apply mask
    mask = torch.ne(inputs, 0)
    outputs = torch.mul(outputs, mask)

    vanilla_MSE = nn.MSELoss()
    reco_loss = vanilla_MSE(inputs.reshape(inputs.shape[0], 57), outputs.reshape(outputs.shape[0], 57))
    return reco_loss

def batch_custom_MSE(inputs, outputs):
    inputs = inputs.reshape((inputs.shape[0], 19, 3, 1))
    outputs = outputs.reshape((outputs.shape[0], 19, 3, 1))

    # trick on phi
    outputs_phi = torch.pi * torch.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0 * torch.tanh(outputs)
    outputs_eta_muons = 2.1 * torch.tanh(outputs)
    outputs_eta_jets = 4.0 * torch.tanh(outputs)
    outputs_eta = torch.cat([outputs[:,0:1,:], outputs_eta_egamma[:,1:5,:], outputs_eta_muons[:,5:9,:], outputs_eta_jets[:,9:19,:]], dim=1)
    outputs = torch.cat([outputs[:,:,0], outputs_eta[:,:,1], outputs_phi[:,:,2]], dim=2)

    # change input shape
    inputs = torch.squeeze(inputs, -1)

    # calculate and apply mask
    mask = torch.ne(inputs, 0)
    outputs = torch.mul(outputs, mask)

    # Reshape inputs and outputs
    inputs = inputs.reshape(inputs.shape[0], 57)
    outputs = outputs.reshape(outputs.shape[0], 57)

    # Calculate MSE per instance in batch
    losses = torch.mean((inputs - outputs)**2, dim=1)

    #print(losses.size())

    return losses

# returns tpr value for specific threshold
# signal_losses is a tensor of individual losses for a given signal dataset
def get_tpr(threshold, signal_losses):
    tpr = torch.sum(signal_losses > threshold).float() / len(signal_losses)
    return tpr

# retruns threshold given an fpr
def get_threshold(fpr, background_losses):
    background_losses = background_losses.sort()[0] # Use PyTorch's sort function and pick values
    num_false_pos = int(len(background_losses) * fpr)
    threshold = background_losses[-num_false_pos]
    return threshold

#exponential linespace because we have a large threshold space to cover
def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)

def get_losses(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    with torch.no_grad():
        for data, data_scaled in dataloader: 
            data, data_scaled = data.float().to(device), data_scaled.float().to(device)
            outputs = model(data)
            losses = batch_custom_MSE(data_scaled, outputs) # SWITCHES
            results.append(losses)
    return torch.cat(results, 0)

def print_tprs(background_losses, list_of_signal_losses, signal_labels, fpr = 1e-5):
    threshold = get_threshold(fpr, background_losses)
    for idx, signal_losses in enumerate(list_of_signal_losses):
      tpr = get_tpr(threshold, signal_losses)
      #print(signal_labels[idx], tpr)
      print(f"{signal_labels[idx]} {tpr.item():.4f}")

def plot_auc(background_losses, list_of_signal_losses, signal_labels, num_fprs=1000, plot=False):
      signal_tprs = torch.zeros(len(list_of_signal_losses), num_fprs)
      fprs = powspace(1e-6, 1, 5, num_fprs) 
      for fpr_idx, fpr in enumerate(fprs):
          threshold = get_threshold(fpr, background_losses)
          for signal_idx, signal_losses in enumerate(list_of_signal_losses):
            tpr = get_tpr(threshold, signal_losses)
            signal_tprs[signal_idx, fpr_idx] = tpr

      # Plot curves for each signal dataset
      for signal_idx in range(len(list_of_signal_losses)):
          plt.plot(fprs, signal_tprs[signal_idx].numpy(), label = signal_labels[signal_idx] + ', auc: {}'.format(( round(auc(fprs, signal_tprs[signal_idx].numpy()), 2) )))
          plt.semilogx()
          plt.semilogy()
          plt.ylabel("True Positive Rate")
          plt.xlabel("False Positive Rate")
          plt.legend(loc='center right')
          plt.grid(True)
          plt.tight_layout()
          plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
          plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
          plt.title("ROC AE")
      plt.show()
