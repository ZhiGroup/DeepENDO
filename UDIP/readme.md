# Generating 128 dimensional UDIPs (also referred to as ENDOs)

## Running the UDIP pipeline

Change the directory to UDIP

```bash
python udip_pipeline.py -i [input] -m [modality] -c [ckpt] -d [device] -o [output]
```
Example
```bash
python udip_pipeline.py -i input.csv -m image_paths -c ckpts/T1.ckpt -d cuda:0 -o output
```

Arguments
- `-i` or `--input`: (required) The input CSV file containing T1 / T2 MRI paths for linearly registered (MNI152 space) brain extracted MRI.
- `-m` or `--modality`: (required) The column name in the CSV file containing the image paths.
- `-c` or `--ckpt`: (required) The path to the checkpoint file.
- `-d` or `--device`: (optional) The device to run on (with a default value of "cuda:0" if not specified).
- `-o` or `--output`: (required) The output directory. Output contains csv containing 128 dimensional UDIPs, losses, and correlation heatmap.


## Code walkthrough pipeline files

udip_dataset.py: Defines custom pytorch dataset

udip_model.py: Defining the model

udip_pipeline.py: Main file

ckpts: Download checkpoints here



## Download ckpts in the ckpts directory

Download the checkpoint file from the provided link and place it in the `ckpt` directory:
     [Checkpoints Download](https://drive.google.com/drive/folders/16IXv-w6xpHhEQiSNjRSI8S5wS4QDjGKE?usp=drive_link)
