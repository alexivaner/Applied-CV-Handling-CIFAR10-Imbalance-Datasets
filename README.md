# Handling CIFAR-10 Imbalance Dataset (Long-Tail Datasets)
This repository is assignment from Applied Computer Vision course in National Chiao Tung University. Please use this repo for learning purposes only. It is far from perfect if you use it for research purpose.

## Methods

### Full Homework Problem
You could see the full problem here:<br>
[Click this link](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/ACV_HW3.pdf)

#### Data Composition
Data training composed as below:<br>
![Data Imbalance](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/Longtail.png)

This repo implement two of many ways to handle imbalance dataset. The code is based on Pytorch. The methods are:<br>

Data test composed with 1000 data for each class or label so the total of data test will be 10000 datas.


#### FIRST METHOD – RESAMPLE THE DATA TRAINING
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling) like what shown below:<br>
![Data Resample](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/Resampling.png)

### SECOND METHOD – APPLY THE LOSS REWEIGHTING
While handling a long-tailed dataset (one that has most of the samples belonging to very few of the classes and many other classes have very little support), deciding how to weight the loss for different classes can be tricky. Often, the weighting is set to the inverse of class support or inverse of the square root of class support. Re-weighting by the effective number of samples gives a better result. The Class-Balanced Loss is designed to address the problem of training from imbalanced data by introducing a weighting factor that is inversely proportional to the effective number of samples. The class-balanced (CB) loss can be written as:

![Equation](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/Equation.png)

## How to run this program:

### From command prompt
I do not upload the checkpoint or my pretrained model here since the size is more than 100 MB and could not uploaded to Github, you could download the pth model here:<br>
[Click this link](https://drive.google.com/file/d/11DDSbPqFXLzooIv6YPmXuKRIZJ24808g/view)<br>
    
Then put the checkpoint model inside `/cifar10_models/state_dicts/` <br>


For default, this repository using DenseNet161, so you could download DenseNet checkpoint, you could change the model also by change Line 111 of 0860812.py"<br>
    `net = densenet161(pretrained=True)`

Thank you for huyvnphan for providing pretrained model for CIFAR-10, I also took model code from his repository, here is his full repository:<br>
[Click this link](https://github.com/huyvnphan/PyTorch_CIFAR10)<br>


You could run the program by following example::<br>
Default (Without resampling and weight rebalancing):<br>
 `python 0860812.py` <br>
Turn on Resampling only: <br>
 `python 0860812.py --resampling_balance True` <br>
Turn on loss reweighting only:<br>
 `python 0860812.py -- reweight_balance True` <br>

## Result
### Accuracy Comparison
![Accuracy Comparison](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/TestAcc.png)

### Conf Matrix Comparison
![Conf Matrix Comparison](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/image/ConfMat.png)

## Conclusion
From the experiment above, we could see that the accuracy graph for both training and test does not change with or without applying loss reweighting or resampling dataset. The overall test accuracy stays around 75%.  But applying reweighting or resampling at a long-tail dataset is important. We could see in the confusion matrices that without apply any balancing effort, our model just learns from class with many datasets only, so our class with few datasets like class 9 will have really bad accuracy at only 0.51.<br><br>
After apply reweighting loss and resample, our overall accuracy did not improve, but we could see that class from a few numbers of datasets have some improvement in terms of accuracy. But we should realize that accuracy in some of the class is also decreased. It is better to make sure the model learns from all the classes in the datasets than keep it learn from classes with many datasets only. We should remember there is no free lunch theory, so apply loss reweighting or rebalancing samples does not mean that the overall accuracy is higher, but it will make sure the model will learn from all the class even from a class with really few datasets.<br><br>
It is not always when we get a high overall accuracy then it means our model is a good one especially when dealing with an imbalanced dataset. What we should try, is to get a higher precision with a higher recall value by examining confusion matrix. But there is often a trade-off between these two metrics. There is a general inverse relation between these two metrics. Meaning if we increase one, more often than not we will decrease the other.


 ## Explanation Report
[Click this link](https://github.com/alexivaner/Applied-CV-Handling-CIFAR10-Imbalance-Datasets/blob/main/Ivan%20Surya%20H_0860812.pdf)

## Reference
Training and tested code are modified from this repository:<br>
 [Pytorch Cifar](https://github.com/kuangliu/pytorch-cifar)<br>

Good Article about Accuracy and Confusion Matrix on Imbalance Datasets:<br>
  [Accuracy on Imbalanced Datasets and Why You Need Confusion Matricex](https://medium.com/analytics-vidhya/accuracy-on-imbalanced-datasets-and-why-you-need-confusion-matrix-937613bf89bf)<br>
  
   [Imbalanced Class](https://elitedatascience.com/imbalanced-classes)<br>







 


