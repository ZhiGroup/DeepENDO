
#imports
from monai import transforms
import pandas as pd
import nibabel as nib
import torch

#defining transforms
transforms_monai = transforms.Compose([transforms.AddChannel(), transforms.ToTensor(),])

class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, modality, transforms):
        
        """

        Args:
            datafile (type: csv): the datafile mentioning the location of images.
            modality (type: string): column containing location of modality of interest in the datafile
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]
        self.transforms = transforms

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idxx=int):
        img = nib.load(self.unbiased_brain[idxx])
        img = img.get_fdata()
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = self.transforms(img)

        img = img.type(torch.float)
        mask = torch.tensor(mask)

        return img, mask

