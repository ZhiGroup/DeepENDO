import torch
from torch import nn

class model_AE(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.hidden_dim = dim
        self.first_cnn = self.first_CNN_block(1, 16)
        self.first_max_poold = self.max_poold((1, 1, 1))
        self.first_encoder = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0, 1, 0))
        self.second_encoder = self.encoder_block(32, 64)
        self.third_max_poold = self.max_poold((1, 0, 1))
        self.third_encoder = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0, 0, 0))
        self.fourth_encoder = self.encoder_block(128, 256)

        self.encoding_mlp = torch.nn.Linear(256 * 12 * 14 * 12, self.hidden_dim)

        self.decoding_mlp = torch.nn.Linear(
            self.hidden_dim, 256 * 12 * 14 * 12
        )  # 128000

        self.first_decoder = self.decoder_block(256, 128)
        self.first_transconv = self.conv_transpose(128, input_padding=(0, 0, 0))
        self.second_decoder = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(64, input_padding=(1, 0, 1))
        self.third_decoder = self.decoder_block(64, 32)
        self.third_transconv = self.conv_transpose(32, input_padding=(0, 1, 0))
        self.fourth_decoder = self.decoder_block(32, 16)
        self.fourth_transconv = self.conv_transpose(16, input_padding=(1, 1, 1))
        self.last_cnn = self.last_CNN_block(16, 1)


    def max_poold(self, max_padding):
        max_pd = nn.MaxPool3d(kernel_size=2, padding=max_padding)
        return max_pd

    def encoder_block(self, input_channels, output_channels, padding=1):
        encoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                output_channels, output_channels, kernel_size=3, padding=padding,
            ),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

        return encoder

    def conv_transpose(self, output_channels, input_padding):
        conv_t = nn.ConvTranspose3d(
            output_channels,
            output_channels,
            kernel_size=2,
            stride=2,
            padding=input_padding,
        )

        return conv_t

    def decoder_block(self, input_channels, output_channels, input_padding=(0, 0, 0)):
        decoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        return decoder

    def first_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                output_channels, output_channels, kernel_size=3, padding=padding,
            ),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

        return cnn_block

    def last_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, output_channels, kernel_size=1),
        )

        return cnn_block

    def forward(self, x):
        x = self.first_cnn(x)
        x = self.first_max_poold(x)
        x = self.first_encoder(x)
        x = self.second_max_poold(x)
        x = self.second_encoder(x)
        x = self.third_max_poold(x)
        x = self.third_encoder(x)
        x = self.fourth_max_poold(x)
        x = self.fourth_encoder(x)
        shape = x.size()

        enc_features = torch.flatten(
            x, start_dim=1, end_dim=-1
        )  

        lin1 = self.encoding_mlp(enc_features)
        dec = self.decoding_mlp(lin1)
        dec = dec.view(shape)
        dec = self.first_decoder(dec)
        dec = self.first_transconv(dec)
        dec = self.second_decoder(dec)
        dec = self.second_transconv(dec)
        dec = self.third_decoder(dec)
        dec = self.third_transconv(dec)
        dec = self.fourth_decoder(dec)
        dec = self.fourth_transconv(dec)
        recon = self.last_cnn(dec)

        return recon, lin1