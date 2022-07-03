import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from modules import AutoEncoder
from modules import visualize_loss
from modules import plot_reconstructed_images
from modules import trainer
from modules import validater
from modules import get_latent_variables
from modules import visualize_latent_variables

torch.manual_seed(0)



def main():
    """
    hyper parameters
    """
    lat_dim = 4
    batch_size = 256
    learning_rate = 0.001
    weight_decay=1e-05
    num_epochs = 40

    # current work-directory path
    cwdpath = Path(os.getcwd())


    """
    Download the MNIST dataset
    """
    dataset_path = 'dataset'
    train_dataset = datasets.MNIST(dataset_path, train=True, download=True)
    test_dataset  = datasets.MNIST(dataset_path, train=False, download=True)

    """
    Pipeline for preprocessing
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # perform preprocessing
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    """
    Split the dataset
    """
    dataset_length = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [int(dataset_length - dataset_length*0.2), int(dataset_length*0.2)])

    """
    Data Loader
    """
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    """
    Initialize an optimizer
    """
    model = AutoEncoder(lat_dim=lat_dim)

    # learning_rate = 0.001
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Pass the objects to the selected device
    model.to(device)

    # Train the model
    train_loss_list, val_loss_list = [], []
    for epoch in range(num_epochs):
        train_loss = trainer(model, train_loader, optim, device=device)
        val_loss = validater(model, test_loader, device=device)
        print('\nEPOCH {}/{}   train loss {:.3f}   val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # plot the original and reconstructed images
        if epoch == num_epochs - 1:
            plot_reconstructed_images(model, test_dataset, 
                                      device=device, num_labels=10,
                                      savepath=cwdpath / 'results' / f"epoch-{epoch+1}_reconstruct_images.png")

    # plot loss at each epoch
    visualize_loss(train_loss_list, val_loss_list, 
                   savepath=cwdpath / 'results' / 'learning_curve.png')

    """
    Postprocess
    """
    # get the latent variables
    lat_vars, labels = get_latent_variables(model, test_dataset, device=device)
    # resize: (10000, 1, 4) -- > (10000, 4)
    lat_vars = lat_vars.reshape(lat_vars.shape[0], lat_vars.shape[2])

    # Visualize the latent variables
    idx_x, idx_y = 0, 1
    visualize_latent_variables(lat_vars, labels, idx_x, idx_y,
                               savepath=cwdpath / 'results' / 'latent_variables.png')


    """
    Dimensional reduction into 2 by t-SNE
    """
    tsne = TSNE(n_components=2)
    lat_vars_tsne = tsne.fit_transform(lat_vars)

    # visualize the latent variables
    idx_x, idx_y = 0, 1
    visualize_latent_variables(lat_vars_tsne, labels, idx_x, idx_y,
                               savepath=cwdpath / 'results' / 'latent_variables_t-SNE.png')



if __name__ == "__main__":
    main()