import numpy as np
import matplotlib.pyplot as plt
import torch



def plot_reconstructed_images(
        model, 
        test_dataset, 
        device='cpu', 
        num_labels=10,
        savepath=None,
    ):
    """
    Visualize reconstructed images

    Args:
        model: AutoEncoder
        dataset(torchvision.datasets.mnist.MNIST): dataset for reconstruction
        device: 'cpu' or 'cuda'
        num_labels(int): number of labels
    Note:
        * imaga size: (1, 1, 28, 28)
            --> imaga.squeeze(): (28, 28)
    """
    labels = test_dataset.targets.numpy()
    test_idx = {i:np.where(labels==i)[0][0] for i in range(num_labels)}

    model.eval()
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(12, 4))
    for i in range(num_labels):
        # original image of i-th-label image, 
        image = test_dataset[test_idx[i]][0].unsqueeze(0).to(device)
        axes[0][i].imshow(image.cpu().squeeze().numpy())

        # reconstruct image
        with torch.no_grad():
            rec_image = model(image)
            rec_image = rec_image.cpu().squeeze().numpy()
        axes[1][i].imshow(rec_image)
    # Set the titles (about) in center position
    axes[0][int(num_labels / 2)].set_title('original images')
    axes[1][int(num_labels / 2)].set_title('reconstructed images')
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close(fig)



if __name__=="__main__":
    pass