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

class SignalDataset(Dataset):
    def __init__(self, signal_file):
        with h5py.File(signal_file, 'r') as file:
            self.data = torch.tensor(np.array(file['scaled_data'])).type(torch.float32) # 'scaled_data'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        return data_sample

class BkgDataset(Dataset):
    def __init__(self, bkg_dataset):
        self.data = bkg_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_dataloaders(path, BATCH_SIZE = 1024):
    bkg_file = path + "all_background_processed.h5"
    signal_files = [
        path + 'all_leptoquark_processed.h5',
        path + 'all_Ato4l_processed.h5',
        path + 'all_hChToTauNu_processed.h5',
        path + 'all_hToTauTau_processed.h5',
    ]
    signal_labels = ['leptoquark','Ato4l','hChToTauNu', 'hToTauTau']


    list_of_signal_dataloaders = [DataLoader(SignalDataset(signal_file),
                                             batch_size= BATCH_SIZE,
                                             shuffle=True)
                                        for signal_file in signal_files]


    # Initialize background data
    with h5py.File(bkg_file, 'r') as file:
            bkg_data = torch.tensor(np.array(file['data_target'])).type(torch.float32) # 'data_target'

    #Split Data
    total_length = len(bkg_data)
    train_boundary = int(0.4 * total_length)
    val_boundary = int(0.5 * total_length)

    bkg_train = bkg_data[:train_boundary]
    bkg_val = bkg_data[train_boundary:val_boundary]
    bkg_test = bkg_data[val_boundary:]
    del(bkg_data)

    bkg_trainloader = DataLoader(BkgDataset(bkg_train), batch_size=BATCH_SIZE, shuffle=True)
    bkg_valloader = DataLoader(BkgDataset(bkg_val), batch_size=BATCH_SIZE, shuffle=False)
    bkg_testloader = DataLoader(BkgDataset(bkg_test), batch_size=BATCH_SIZE, shuffle=False)

    return list_of_signal_dataloaders, bkg_trainloader, bkg_valloader, bkg_testloader, signal_labels

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size=57, latent_dim=3):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            self._init_weights(nn.Linear(input_size, 32)),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            self._init_weights(nn.Linear(32, 16)),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3)
        )
        self.fc_mu = self._init_weights(nn.Linear(16, latent_dim))
        self.fc_logvar = self._init_weights(nn.Linear(16, latent_dim))

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

    def reparameterize(self, mu, logvar):
        batch, dim = mu.shape
        epsilon = torch.randn(batch, dim).to(mu.device)  # ensure epsilon is on the same device as mu
        return mu + torch.exp(0.5 * logvar) * epsilon


    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        #print(logvar)

        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar, z


    def _init_weights(self, layer):
        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer

def kl_divergence(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.mean(kl, dim=-1)  # Take the mean only across the latent dimension
    return kl

def train(model, train_loader, num_epochs, lr = 0.001, early_stopping_patience=10, get_losses = False, val_loader = None, beta = 0.8):
    # Init training optimizers
    optimizer = optim.Adam(model.parameters(), lr = lr)
    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True, min_lr=1E-6)
    #criterion = nn.MSELoss()
    criterion = custom_MSE

    best_model_state = None  # Save the best model's state dict here

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    device = next(model.parameters()).device # added for cuda

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.float().to(device)  # ensure data is on the same device as the model (added for cuda)
            optimizer.zero_grad()

            # Here, we get both outputs, mu, and logvar from the model
            outputs, mu, logvar, _ = model(data)

            # Calculate the reconstruction loss
            reconstruction_loss = criterion(outputs, data)
            # Calculate the KL divergence, and normalize it by the batch size
            kl = kl_divergence(mu, logvar)

            #print(kl)
            # The total loss is a weighted sum of the reconstruction and KL divergence loss
            loss = (1-beta) * reconstruction_loss + beta * kl.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
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
                best_model_state = copy.deepcopy(model.state_dict())  # Update best model state
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered after {} epochs".format(epoch + 1))
                model.load_state_dict(best_model_state)  # Revert to best model state
                break

    if get_losses == True:
        return train_losses, val_losses

def test(model, dataloader, beta = 0.8):
    # Testing the autoencoder
    model.eval()
    criterion = custom_MSE
    test_loss = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.float().to(device)  # Move data to the device that the model is on

            # We now expect the model to return reconstructions, mu, and logvar
            outputs, mu, logvar, _ = model(data)

            # Calculate the reconstruction loss
            reconstruction_loss = criterion(outputs, data)

            # Calculate the KL divergence, and normalize it by the batch size
            kl = kl_divergence(mu, logvar)

            # The total loss is a weighted sum of the reconstruction and KL divergence loss
            loss = (1-beta) * reconstruction_loss + beta * kl.sum()
            test_loss += loss.item()

    test_loss = test_loss / len(dataloader)
    print('Loss: {:.4f}'.format(test_loss))
    return test_loss

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

    vanilla_MSE = nn.MSELoss(reduction='sum')
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
        for data in dataloader:
            data = data.float().to(device)
            reconstructions, _, _, _ = model(data)  # Get reconstructions only (discard mu and logvar)
            losses = batch_custom_MSE(reconstructions, data)
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
