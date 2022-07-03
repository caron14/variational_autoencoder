import numpy as np
import torch



def trainer(model, dataloader, optimizer, device='cpu'):
    """
    Train the model

    Args:
        model: AutoEncoder
        device: 'cpu' or 'cuda'
    Returns:
        Average loss
    """
    # switch to the train mode
    model.train()
    
    # train the model
    loss_mean = []
    for images, _ in dataloader:
        images = images.to(device)
        # reconstruct images
        rec_images = model(images)
        # calculate loss
        loss = torch.nn.MSELoss()(rec_images, images)
        # update the model parameters
        optimizer.zero_grad(optimizer)
        loss.backward()
        optimizer.step()
        # 
        loss_mean.append(loss.detach().cpu().numpy())
    
    return np.mean(loss_mean)


def validater(model, dataloader, device='cpu'):
    """
    Evaluate a validation loss

    Args:
        model: AutoEncoder
        device: 'cpu' or 'cuda'
    Returns:
        Average loss
    """
    # switch to the evaluation model
    model.eval()
    # without info. of the gradients
    original_images, reconstructed_images = [], []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # reconstruct images
            rec_images = model(images)
            # store the original and reconstructed images
            original_images.append(images.cpu())
            reconstructed_images.append(rec_images.cpu())
        # concatenate the components of list
        original_images = torch.cat(original_images)
        reconstructed_images = torch.cat(reconstructed_images)
        # calculate the loss
        loss = torch.nn.MSELoss()(reconstructed_images, original_images)
    
    return loss.detach().cpu().numpy()



if __name__=="__main__":
    pass