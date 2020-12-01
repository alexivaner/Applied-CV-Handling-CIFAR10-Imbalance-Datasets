# Handling CIFAR-10 Imbalance Dataset (Long-Tail Datasets)
This repository is assignment from Applied Computer Vision course in National Chiao Tung University. Please use this repo for learning purposes only. It is far from perfect if you use it for research purpose.

## Methods

### Full Homework Problem
You could see the full problem here:<br>
[Click this link](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/ACV_HW3.pdf)

This repo implement two of many ways to handle imbalance dataset. The code is based on Pytorch. The methods are:<br>

#### Data Composition
Data training composed as below:<br>
![Data Imbalance](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/Longtail.png)

Data test composed with 1000 data for each class or label so the total of data test will be 10000 datas.


#### FIRST METHOD – RESAMPLE THE DATA TRAINING
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling) like what shown below:<br>
![Data Resample](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/Resampling.png)

### SECOND METHOD – APPLY THE LOSS REWEIGHTING
While handling a long-tailed dataset (one that has most of the samples belonging to very few of the classes and many other classes have very little support), deciding how to weight the loss for different classes can be tricky. Often, the weighting is set to the inverse of class support or inverse of the square root of class support. Re-weighting by the effective number of samples gives a better result. The Class-Balanced Loss is designed to address the problem of training from imbalanced data by introducing a weighting factor that is inversely proportional to the effective number of samples.

## How to run this program:

### From command prompt
I do not upload the checkpoint or my pretrained model here since the size is more than 100 MB and could not uploaded to Github, you could download the pth model here:<br>
[Click this link](https://drive.google.com/file/d/11DDSbPqFXLzooIv6YPmXuKRIZJ24808g/view)<br>


Then put the checkpoint model inside `/cifar10_models/state_dicts/` <br>

Thank you for huyvnphan for providing pretrained model for CIFAR-10, I also took model code from his repository, here is his full repository:<br>
[Click this link](https://github.com/huyvnphan/PyTorch_CIFAR10)<br>


You could run the program by following example::<br>
Default (Without resampling and weight rebalancing):<br>
 `python 0860812.py`
Turn on Resampling only:<br>
 `python 0860812.py --resampling_balance True`
Turn on loss reweighting only:<br>
 `python 0860812.py -- reweight_balance True`

## Result
### Image Pyramid
![Gaussian Pyramid](https://github.com/alexivaner/Image-Matching-using-SIFT-and-FLANN-Method/blob/main/result/gaussian_image.png)

### Image DoG
![DoG Images](https://github.com/alexivaner/Image-Matching-using-SIFT-and-FLANN-Method/blob/main/result/dog_image.png)

### Image KeyPoint
![Image Keypoints](https://github.com/alexivaner/Image-Matching-using-SIFT-and-FLANN-Method/blob/main/result/image_keypoints.png)


 ## Explanation Report
[Click this link](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/Ivan%20Surya%20H_0860812.pdf)

## Reference
This code is combined and modified from this source:<br>
 [rmIslam PythonSIFT](https://github.com/rmislam/PythonSIFT)<br>
 Thank you for your clear code about how to find extrema keypoint.

Good Article about Accuracy and Confusion Matrix on Imbalance Datasets:<br>
  [Accuracy on Imbalanced Datasets and Why You Need Confusion Matricex](https://medium.com/analytics-vidhya/accuracy-on-imbalanced-datasets-and-why-you-need-confusion-matrix-937613bf89bf)<br>
  






 


