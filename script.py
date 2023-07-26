from methods import *

path = 'autoencoder-prune/'
list_of_signal_dataloaders, bkg_train_loader, bkg_val_loader, bkg_test_loader, signal_labels = get_data_loaders(path, BATCH_SIZE)

# Run pruning script
model = Autoencoder()

# Store returned losses and other metrics
train_losses, val_losses, zeros, total, tprs, background_losses_all, signal_losses_all = Prune(model, bkg_train_loader, bkg_val_loader, pruning_iters = 5, num_epochs = 100, amount = .12, delay_pruning_epochs=0)

sparsity, train_losses_last, val_losses_last = plot_and_return_losses_vs_sparsity(zeros, total, train_losses, val_losses, plot=False)

tprs_all_datasets = plot_and_return_tprs(tprs, plot=False)

fpr_losses, tpr_losses = calc_and_plot_losses(background_losses_all, signal_losses_all, signal_labels, num_thresholds=1000, plot=False)

losses, labels = plot_histogram(background_losses_all, signal_losses_all, signal_labels, bin_size=100, plot=False)

experiment_number = 1
variables_to_save = ["train_losses", "val_losses", "zeros", "total", "tprs",
                     "background_losses_all", "signal_losses_all", "sparsity",
                     "train_losses_last", "val_losses_last", "tprs_all_datasets",
                     "fpr_losses", "tpr_losses", "losses"]

for var in variables_to_save:
    np.save(f'{var}_{experiment_number}.npy', locals()[var])
