import torch
from torch import nn



class Encoder(nn.Module):
    """
    Encoder for 2-dimensional Convolutional AutoEncoder
    """
    def __init__(self, lat_dim) -> None:
        super().__init__()
        # convolutional part
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # dense layer
        self.encoder_dense = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, lat_dim)
        )
    
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_dense(x)
        return x


class Decoder(nn.Module):
    """
    Decoder for 2-dimensional Convolutional AutoEncoder
    """
    def __init__(self, lat_dim) -> None:
        super().__init__()
        # dense part
        self.decoder_dense = nn.Sequential(
            nn.Linear(lat_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        # unflatten part
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        
        # convolution part
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        x = self.decoder_dense(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    """
    2-dimensional Convolutional AutoEncoder
    """
    def __init__(self, lat_dim) -> None:
        super().__init__()
        self.encoder = Encoder(lat_dim=lat_dim)
        self.decoder = Decoder(lat_dim=lat_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__=="__main__":
    pass