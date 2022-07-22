
#Prostate Gleason Classifiaction with Fuse

**Introduction**

The project presents lesions classification of Gleason score in prostate. It demonstrates the use of FuseMedML for classification of 3D lesions extracted from multi-MRI which includes several types of series imaging. The classification is binary between high Gleason lesions (>=7) and low Gleason lesions (<7).

**Dataset**

We used the public SPIE-AAPM-NCI PROSTATEx Challenge dataset: 
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656

**Pre-processing**

The pre-processing is preformed using two major functions: processor_dicom_mri.py and processor.py. The processor_dicom_mri.py extract the relevant MRI sequences and stack them together, registrated by world coordinates system and resampled with common resolution. Then, the processor.py crops the subvolume of the image based on the given lesion mask volume. The sequences that are in use in this example are (T2, ADC, DWI(b), Ktrans). The output of the pre-processing is 4d volume of size (MRI series(ch),z,x,y) around lesion location point.

**CNN**

This project is an implementation on the 3D ResNet as was presented at "Semi-automatic classification of prostate cancer on multi-parametric MR imaging using a multi-channel 3D convolutional neural network".

**INPUT**

The network is trained using cross-validation. In each iteration one fold ('fold_no') is set as validation and the other folds as training data. The different folds are extracted from dataset.pickle. The folds data and labels are extracted from Lesion Information inside ProstateX-Finding-Train.csv that can be found in the challenge's site.

Two types of data paths are available:

    * prostate_data_path - which contains PROSTATEx folder as was downloaded from the challenge's site.
    * ktrans_path - which contains Ktrans images as were downloaded from the challenge's site (see Detailed Description)

**MODEL TRAINING**

    * In run_train_3dpatch.py:
        fill in the code with paths that are labeled as TODO
        specify fold_no to run
    * Run: python3 run_train_3dpatch.py

**Results**

For baseline comparison to paper results a run using only one of the MRI series (Ktrans) was performed. The average AUC score of the 8 folds cross-validation was 0.8 (similarly to the paper). When using the full series as input (T2,DWI(b=400),DWI(b=800),ADC,Ktrans) the average AUC was 0.83. This result is slightly lower than the result in the paper since no registration between series was performed (should be added to pre-processing).
