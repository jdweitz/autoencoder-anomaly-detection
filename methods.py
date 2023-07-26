import numpy as np
import h5py
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

class SignalDataset(Dataset):
    def __init__(self, signal_files, signal_labels): #signals is a list of all the signal datasets you want to include
        signal_data = []
        for i, label in enumerate(signal_labels):
            with h5py.File(signal_files[i], 'r') as file:
                test_data = np.array(file['Data'])
            signal_data.append(test_data)

        self.data = np.squeeze(np.array(signal_data)) # save signal data as np array


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index])

class BkgDataset(Dataset):
    def __init__(self, bkg_file, split): #split = 'X_train', or 'X_test', etc.
        with h5py.File(bkg_file, 'r') as file:
          self.data = torch.tensor(file[split][:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_data_loaders(path, BATCH_SIZE = 1024):
    bkg_file = path + 'BKG_dataset.h5'
    signal_1 = path + 'Ato4l_lepFilter_13TeV_dataset.h5'
    signal_2 = path + 'hChToTauNu_13TeV_PU20_dataset.h5'
    signal_3 = path + 'hToTauTau_13TeV_PU20_dataset.h5'
    signal_4 = path + 'leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5'

    # add correct signal labels
    signal_labels = ['Ato4l','hChToTauNu', 'hToTauTau', 'leptoquark']

    # add correct path to signal files
    signal_files = [signal_1, signal_2, signal_3, signal_4]

    # create datasets
    list_of_signal_dataloaders = []
    for i in range(4):
      #init datasets
      dataset = SignalDataset([signal_files[i]],[signal_labels[i]])
      #make loader
      dataloader = DataLoader(dataset, batch_size=1024, shuffle=True) # different than actual batch size
      #add loader to list
      list_of_signal_dataloaders.append(dataloader)

    # Initialize background data
    bkg_train_loader = DataLoader( BkgDataset(bkg_file,'X_train'), batch_size=BATCH_SIZE, shuffle=True)
    bkg_val_loader = DataLoader( BkgDataset(bkg_file,'X_val'), batch_size=BATCH_SIZE, shuffle=False)
    bkg_test_loader = DataLoader( BkgDataset(bkg_file,'X_test'), batch_size=BATCH_SIZE, shuffle=False)

    return list_of_signal_dataloaders, bkg_train_loader, bkg_val_loader, bkg_test_loader, signal_labels

class Autoencoder(nn.Module):
    def __init__(self, input_size = 57):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 3),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 57)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, train_loader, num_epochs, lr = 0.001, early_stopping_patience=10, get_losses = False, val_loader = None):
    # Init training optimizers
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            outputs = model(data.float())
            loss = criterion(outputs, data.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('epoch [{}/{}], Training loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_loader)))
        if get_losses == True:
            val_loss = test(model, val_loader)
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)

            #print('epoch [{}/{}], Validation loss:{:.4f}'.format(epoch + 1, num_epochs, val_loss))

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
  criterion = nn.MSELoss()
  test_loss = 0
  with torch.no_grad():
      for data in dataloader:
          outputs = model(data.float())
          loss = criterion(outputs, data.float())
          test_loss += loss.item()

  test_loss = test_loss / len(dataloader)

  print('Loss: {:.4f}'.format(test_loss))
  return test_loss

def get_tpr(fpr, background_losses, signal_losses): #signal_losses is just the losses for one specifc signal dataset
  background_losses.sort()
  num_false_pos = int(len(background_losses) * fpr)
  threshold = background_losses[-num_false_pos]
  tpr = np.sum(signal_losses > threshold) / len(signal_losses)
  return tpr

def compute_losses(model, dataloader_test, list_of_signal_dataloaders):
    def batch_MSE(outputs, targets):
        loss = torch.mean((outputs - targets)**2, axis = 1)
        return loss

    #Get losses for different signal datasets
    model.eval()

    # Compute losses for the background dataset
    background_losses = []
    with torch.no_grad():
        for data in dataloader_test:  # using the validation dataloader you defined
            outputs = model(data.float())
            loss = batch_MSE(outputs, data.float())
            background_losses += list(loss)

    background_losses = np.array(background_losses)

    # Compute losses for different signal datasets
    signal_losses = []
    for i in range(4):
        losses = [] #[1,2,3,...] [[1,2,3,4],[5,6,7,8],...]
        with torch.no_grad():
            for data in list_of_signal_dataloaders[i]:
                outputs = model(data.float())
                loss = batch_MSE(outputs, data.float())
                losses += list(loss)
        signal_losses.append(np.array(losses))

    return background_losses, signal_losses

#exponential linespace because we have a large threshold space to cover
def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)

def get_parameters_to_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)[1:]

def sparsity_print(model):
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    zero = total = 0
    for module, _ in get_parameters_to_prune(model):
        zero += float(torch.sum(module.weight == 0))
        total += float(module.weight.nelement())
    print('Number of Zero Weights:', zero)
    print('Total Number of Weights:', total)
    print('Sparsity', zero/total)
    return zero, total

def Prune(model, train_loader, val_loader, pruning_iters = 30, num_epochs = 60, amount = .2, delay_pruning_epochs = 0):
    zeros = [] #keeps track of zeros at each iteration
    tprs = [] #keeps track of tpr at each pruning iteration
    train_losses = [] #keeps track of training loss at each pruning iteration
    val_losses = [] #keeps track of validation loss at each pruning iteration
    background_losses_all = [] #keeps track of background losses at each pruning iteration
    signal_losses_all = [] #keeps track of signal losses at each pruning iteration
    fpr_target = 0.00001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Create rewind point for model
    model_rewind = copy.deepcopy(model).to(device)

    #Pretrain model before pruning
    train_loss, val_loss = train(model, train_loader, num_epochs = num_epochs, get_losses = True, val_loader = val_loader)

    #Lottery Ticket Rewinding: Prune, Rewind, Train
    for i in range(pruning_iters):
        print('Pruning Iteration:', i+1)
        #Prune
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=amount)
        #Rewind Weights
        for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
            with torch.no_grad():
                module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                module.weight_orig.copy_(module_rewind.weight)
        #Train Weights
        train_loss, val_loss = train(model, train_loader, num_epochs = num_epochs, get_losses = True, val_loader = val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Compute background and signal losses after training
        background_losses, signal_losses = compute_losses(model, bkg_test_loader, list_of_signal_dataloaders)
        background_losses_all.append(background_losses)
        signal_losses_all.append(signal_losses)

        #Log Results
        zero, total = sparsity_print(model)
        zeros.append(zero)

        # Compute TPR for specific FPR
        tpr_iter = [] #keeps track of tpr for each signal dataset in this iteration
        for j in range(4):
            tpr = get_tpr(fpr_target, background_losses, signal_losses[j])
            tpr_iter.append(tpr)
            print(f'Pruning Iteration {i+1}, Signal Dataset {j+1}: TPR at FPR {fpr_target}: {tpr}')
        tprs.append(tpr_iter)

    return train_losses, val_losses, zeros, total, tprs, background_losses_all, signal_losses_all

def plot_and_return_losses_vs_sparsity(zeros, total, train_losses, val_losses, plot=False):
    # Convert zeros to sparsity (divide by total)
    sparsity = [zero / total for zero in zeros]

    # Create lists to return
    train_losses_last = [losses[-1] for losses in train_losses]
    val_losses_last = [losses[-1] for losses in val_losses]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss vs. Sparsity")
        plt.plot(sparsity, train_losses_last, label="train")
        plt.plot(sparsity, val_losses_last, label="val")
        plt.xlabel("Sparsity")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return sparsity, train_losses_last, val_losses_last

def plot_and_return_tprs(tprs, plot=False):
    # Number of signal datasets
    num_datasets = len(tprs[0])

    # Add correct signal labels
    signal_labels = ['Ato4l','hChToTauNu', 'hToTauTau', 'leptoquark']

    tprs_all_datasets = []
    # For each signal dataset
    for i in range(num_datasets):
        # Extract the TPR for each pruning iteration for the current signal dataset
        tprs_dataset = [tprs_iter[i] for tprs_iter in tprs]
        tprs_all_datasets.append(tprs_dataset)
        if plot:
            plt.plot(tprs_dataset, label=signal_labels[i]) # Use the actual signal names from the list

    if plot:
        # Set the labels for the x and y axes
        plt.semilogy()
        plt.xlabel('Pruning Iteration')
        plt.ylabel('TPR')

        # Set the title for the plot
        plt.title('TPR at each Pruning Iteration')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
        
    return tprs_all_datasets

def calc_and_plot_losses(background_losses, signal_losses, signal_labels, num_thresholds=1000, plot=False):
  fpr_losses_all = []
  tpr_losses_all = []
  for pruning_iter in range(len(background_losses)): #number of pruning iters
      fpr_losses = []
      tpr_losses = np.zeros((4, num_thresholds))

      # For each possible threshold, calculate the tpr/fpr loss
      for idx, threshold in enumerate(powspace(0,500000, 10, num_thresholds)): #fix later to smaller increments
          # Calculate fpr from the background data
          num_false_pos = np.sum(background_losses[pruning_iter] > threshold)
          fpr_loss = num_false_pos / len(background_losses[pruning_iter]) # fpr = fp/(fp+tn)
          fpr_losses.append(fpr_loss)

          # Calculate tpr for each signal dataset
          for i in range(4):
              losses = signal_losses[pruning_iter][i]
              num_true_pos = np.sum(losses > threshold) # counts number of individual losses above threshold
              tpr_loss = num_true_pos / len(losses)
              tpr_losses[i,idx] = tpr_loss

      if plot:
          # Plot curves for each signal dataset
          for i in range(4):
              plt.plot(fpr_losses, tpr_losses[i], label = signal_labels[i] + ', auc: {}'.format(( round(auc(fpr_losses,tpr_losses[i]),2) )))
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

      fpr_losses_all.append(fpr_losses)
      tpr_losses_all.append(tpr_losses)

  return fpr_losses_all, tpr_losses_all

def plot_histogram(background_losses, signal_losses, signal_labels, bin_size=100, plot=False):
    # Preparing data and labels for the plot
    losses = [background_losses] + list(signal_losses)
    labels = ['background'] + [signal_labels[i] for i in range(4)]
    
    if plot:
        plt.figure(figsize=(10,8))
        for i in range(5):
            plt.hist(losses[i], bins=bin_size, density=True, histtype='step', fill=False, linewidth=1.5, label = labels[i])

        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel("Autoencoder Loss")
        plt.ylabel("Probability (a.u.)")
        plt.title('MSE loss')
        plt.legend()  # Show legend to distinguish different histograms
        plt.show()

    return losses, labels

