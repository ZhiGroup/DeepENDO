#please use affine registered MRI. Instructions in /training/readme.md

#imports
from monai import transforms
import pandas as pd
import nibabel as nib
import torch

#defining transforms
transforms_monai = transforms.Compose([transforms.AddChannel(), transforms.ToTensor()])

class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, modality, transforms=transforms_monai):
        """
        Args:
            datafile (type: csv or list): the datafile mentioning the location of images or a list of file locations.
            modality (type: string): column containing location of modality of interest in the datafile.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
            img_name [string]: name of the image
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]

        self.transforms = transforms

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idxx=int):
        img_name = self.unbiased_brain[idxx]
        img = nib.load(img_name)
        img = img.get_fdata()
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = self.transforms(img)

        img = img.type(torch.float)
        mask = torch.tensor(mask)

        return img, mask, img_name
