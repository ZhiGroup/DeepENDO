import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from udip_dataset import *
from udip_model import *
from tqdm import tqdm
import numpy as np
import seaborn as sns

def run_pipeline(datafile, modality, ckpt, device, output_dir):
    modality = str(modality)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_T1 = model_AE(128)
    model_T1 = model_T1.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model_T1.load_state_dict(checkpoint["state_dict"])
    T1_ds = aedataset(
        datafile=datafile,
        modality=modality,
    )
    val_dataloader = torch.utils.data.DataLoader(
        T1_ds, batch_size=1, pin_memory=True, num_workers=4, shuffle=False
    )

    loss_fn = torch.nn.MSELoss(reduction="none")

    model_T1 = model_T1.eval()
    losses = []
    UDIPs = []
    image_names = []
    with torch.no_grad():
        for data in tqdm(val_dataloader, total=len(val_dataloader)):
            img, mask, img_name = data
            img = img.to(device)
            mask = mask.to(device)
            recon, lin1 = model_T1(img)
            loss = loss_fn(img, recon)
            loss1 = loss.squeeze(1) * mask
            loss2 = loss1.sum()
            loss3 = loss2 / mask.sum()
            losses.append(loss3.cpu().numpy())
            UDIPs.extend(lin1.cpu().numpy())
            image_names.extend(img_name)
        
    UDIP_df = pd.DataFrame(UDIPs)

    correlation = abs(UDIP_df .corr())

    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlation, cmap="YlGnBu", vmin=0, vmax=1)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "correlation_heatmap.png")

    plt.savefig(output_file)

    mean_losses = np.mean(np.array(losses))
    std_losses_val_64 = np.std(np.array(losses))

    print("The mean of the losses is: {:.4f}".format(mean_losses))
    print("The standard deviation of the losses_val_64 is: {:.4f}".format(std_losses_val_64))

    # Save losses and standard deviation to a text file
    with open(os.path.join(output_dir, 'losses.txt'), 'w') as f:
        f.write("The mean of the losses is: {:.4f}\n".format(mean_losses))
        f.write("The standard deviation of the losses_val_64 is: {:.4f}\n".format(std_losses_val_64))

    # Save endos dataframe to a csv file
    UDIP_df.to_csv(os.path.join(output_dir, 'UDIP.csv'), index=False)
    
    print("Run completed successfully. The output files are saved at {}".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline with specified parameters.')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-m', '--modality', required=True, help='Column name of input csv file containing the image paths')
    parser.add_argument('-c', '--ckpt', required=True, help='Path to the checkpoint file')
    parser.add_argument('-d', '--device', default="cuda:0", help='Device to run on, eg: cuda:0 or cpu')
    parser.add_argument('-o', '--output', required=True, help='Output directory')

    args = parser.parse_args()

    run_pipeline(args.input,args.modality,  args.ckpt, args.device, args.output)
