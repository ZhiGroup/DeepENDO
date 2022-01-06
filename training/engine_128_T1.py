# imports

# PyTorch
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Lightning

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Custom imports
from dataset import *

# Model architecture and forward pass to Pytorch lightning module.

class engine_AE(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = 128  # hidden dimension for latent space used as endophenotype

        # defining layers
        # first CNN block
        self.first_cnn = self.first_CNN_block(1, 16)

        # encoders
        self.first_max_poold = self.max_poold((1, 1, 1))
        self.first_encoder = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0, 1, 0))
        self.second_encoder = self.encoder_block(32, 64)
        self.third_max_poold = self.max_poold((1, 0, 1))
        self.third_encoder = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0, 0, 0))
        self.fourth_encoder = self.encoder_block(128, 256)

        # latent space
        self.encoding_mlp = torch.nn.Linear(256 * 12 * 14 * 12, self.hidden_dim)

        self.decoding_mlp = torch.nn.Linear(self.hidden_dim, 256 * 12 * 14 * 12)

        # decoders
        self.first_decoder = self.decoder_block(256, 128)
        self.first_transconv = self.conv_transpose(128, input_padding=(0, 0, 0))
        self.second_decoder = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(64, input_padding=(1, 0, 1))
        self.third_decoder = self.decoder_block(64, 32)
        self.third_transconv = self.conv_transpose(32, input_padding=(0, 1, 0))
        self.fourth_decoder = self.decoder_block(32, 16)
        self.fourth_transconv = self.conv_transpose(16, input_padding=(1, 1, 1))

        # last CNN block
        self.last_cnn = self.last_CNN_block(16, 1)

        # loss function to be used in training loop
        self.train_loss_function1 = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction="none"
        )
        # loss function to be used in validation loop
        self.valid_loss_function = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction="none"
        )

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

    # Forward function
    def forward(self, x):
        x = self.first_cnn(x)  # 1,16,182,218,182
        x = self.first_max_poold(x)  # 1,16,92,110,92
        x = self.first_encoder(x)  # 1,32,92,110,92
        x = self.second_max_poold(x)  # 1,32,46,56,46
        x = self.second_encoder(x)  # 1,64,46,56,46
        x = self.third_max_poold(x)  # 1,64,24,28,24
        x = self.third_encoder(x)  # 1,128,24,28,24
        x = self.fourth_max_poold(x)  # 1,128,12,14,12
        x = self.fourth_encoder(x)  # 1,256,12,14,12
        shape = x.size()

        # flattening encoder output
        enc_features = torch.flatten(
            x, start_dim=1, end_dim=-1
        )  # to keep batch dimension intact

        lin1 = self.encoding_mlp(enc_features)  # 1,128
        # Going from hidden dimension to original image recon
        dec = self.decoding_mlp(lin1)  # 1,516096
        dec = dec.view(shape)  # 1,256,12,14,12
        dec = self.first_decoder(dec)  # 1,128,12,14,12
        dec = self.first_transconv(dec)  # 1,128,24,28,24
        dec = self.second_decoder(dec)  # 1,64,24,28,24
        dec = self.second_transconv(dec)  # 1,64,46,56,46
        dec = self.third_decoder(dec)  # 1,32,46,56,46
        dec = self.third_transconv(dec)  # 1,32,92,110,92
        dec = self.fourth_decoder(dec)  # 1,16,92,110,92
        dec = self.fourth_transconv(dec)  # 1,16,182,218,182
        recon = self.last_cnn(dec)  # 1, 182, 218, 182

        return recon, lin1

    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        # x, reg_input = batch
        x, mask = batch
        recon, _ = self(x)

        loss1 = self.train_loss_function1(x, recon)
        loss1 = loss1.squeeze(1) * mask
        loss1 = loss1.sum()
        loss1 = loss1 / mask.sum()
        # loss2 = self.train_loss_function(reg_input, reg)
        # loss = loss1 + loss2
        self.log("train_loss", loss1, prog_bar=True)
        return loss1

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        recon, _ = self(x)
        loss1 = self.valid_loss_function(x, recon)
        loss1 = loss1.squeeze(1) * mask
        loss1 = loss1.sum()
        loss1 = loss1 / mask.sum()
        # loss2 = self.train_loss_function(reg_input, reg)
        # loss = loss1 + loss2
        self.log("val_loss", loss1, prog_bar=True, sync_dist=True)
        return loss1

    # pytorch lightning optimizer configuration
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=4,
                min_lr=self.hparams["lr"] / 1000,
                factor=0.5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


# defining train dataset
train_dataset = aedataset(
    datafile="train.csv", modality="T1_unbiased_linear", transforms=transforms_monai,
)

# defining train dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=36, pin_memory=True, num_workers=12, shuffle=True,
)


# defining validation dataset
val_dataset = aedataset(
    datafile="validation.csv",
    modality="T1_unbiased_linear",
    transforms=transforms_monai,
)

# defining validation dataloader
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=36, pin_memory=True, num_workers=12, shuffle=False
)

# directory name to save checkpoints and metrics
dir_name = "T1_128"

# initiaing the model
AE_model = engine_AE(0.0002511886431509582)

# learning rate monitor as using scheduler
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# saving checkpoints monitoring validation loss
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{train_loss:.6f}-{val_loss:.6f}",
    save_top_k=5,
)

# Loggers
tb_logger = TensorBoardLogger(save_dir=dir_name + "/tb_logs")
csv_logger = CSVLogger(save_dir=dir_name + "/csv_logs")
pb = ProgressBar(refresh_rate=2)

# main training
if __name__ == "__main__":
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        # Change the number of GPUs here
        gpus=[0, 1, 2, 3],
        callbacks=[lr_monitor, model_checkpoint, pb],
        sync_batchnorm=True,
        log_every_n_steps=20,
        accelerator="dp",
        benchmark=True,
        max_epochs=100,
    )

    trainer.fit(
        AE_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
