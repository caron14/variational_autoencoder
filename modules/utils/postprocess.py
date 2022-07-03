import numpy as np
import matplotlib.pyplot as plt
import torch



def get_latent_variables(model, dataset, device='cpu'):
    """
    Obtain latent variables

    Args:
        model: trained AutoEncoder model
        test_dataset(torchvision.datasets.mnist.MNIST): test dataset
        device: 'cpu' or 'cuda:0'
    Returns:
        latent variable(ndarray)
        labels(ndarray)
    """
    lat_vars_list, label_list = [], []
    for image, label in dataset:
        image = image.to(device)
        with torch.no_grad():
            # Change size: (1, 28, 28) --> (1, 1, 28, 28)
            lat_vars = model.encoder(image.view(1, 1, image.size(1), image.size(1)))
        # Store the latent variable and label
        lat_vars_list.append(lat_vars.detach().cpu().numpy())
        label_list.append(label)
    
    return np.array(lat_vars_list), np.array(label_list)


def visualize_latent_variables(lat_vars, labels, idx_x, idx_y, savepath=None):
    """
    Visualize the latent variables

    Args:
        lat_vars(ndarray): latent variables
        labels(ndarray): labels(0 ~ 9) of latent variables
        idx_x(int): index of the latent variables
        idx_y(int): index of the latent variables
    """
    fig = plt.figure(figsize=(8, 8))
    for label in set(labels):
        _lat_vars = lat_vars[labels == label]
        plt.scatter(_lat_vars[:, idx_x], _lat_vars[:, idx_y], label=str(label))
    plt.xlabel(f"var {idx_x}", fontsize=14)
    plt.ylabel(f"var {idx_y}", fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close(fig)



if __name__=="__main__":
    pass