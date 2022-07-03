import matplotlib.pyplot as plt



def visualize_loss(train_loss_list, val_loss_list, savepath=None):
    """
    plot the loss at each epoch

    Args:
        train_loss_list(list): training loss at each apoch
        val_loss_list(list): validation loss at each apoch
    """
    fig = plt.figure(figsize=(8, 8))
    plt.semilogy(train_loss_list, label='train', color='red')
    plt.semilogy(val_loss_list, label='valid', color='blue')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('average Loss', fontsize=14)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close(fig)



if __name__=="__main__":
    pass