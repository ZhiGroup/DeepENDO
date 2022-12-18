## Code walkthrough training files

- dataset.py: Defines custom pytorch dataset
- engine_128_T1.py: Code for training using T1 modality
- engine_128_T2.py: Code for training using T2 modality

<hr />

Dataset file needs a csv mentioning the paths of T1 and T2 MRI images. CSV should contain EID, T1_unbiased_linear, T2_unbiased_linear fields where EID is the patient ID, T1_unbiased_linear contains file path for T1 unbiased linearly registered brain extracted images and T2_unbiased_linear contains file path for T2 unbiased linearly registered brain extracted images.

We use FSL 6.0.5 for affine registration of brain extracted MRI provided by UK Biobank using precomputed affine matrix. 

The syntax is as follows:
flirt [options] -in <inputvol> -ref <refvol> -applyxfm -init <matrix> -out <outputvol>

An example syntax using UK Biobank provided directory structure will be as follows:
/fsl/bin/flirt -in /T1/T1_unbiased_brain.nii.gz -applyxfm -init /T1/transforms/T1_to_MNI_linear.mat -out /T1_unbiased_brain_linear.nii.gz -paddingsize 0.0 -interp trilinear -ref /fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

