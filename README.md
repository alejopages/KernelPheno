# KernelPheno
Kernel Phenotyping project management and computational pipelines

## TODO:

Determining background pixel means:

* plot bar plot of pixel distribution
* plot mean selected by kmeans algorithm

Re-evaluting training dataset annotations:
* take all grayscale annotated images and pair them with their RGB images. 
* Re-upload to zooniverse and have leandra validate the annotations

Training:
  CNN:
    We need to determine if we have enough training data after data augmentation
1. if training and validation loss does not decrease we do not have sufficient data
2. Model the performance of the model with respect to the number of training examples
3. Larger number of nuerons and other hyperparameters requires more data to train
4. SOLUTION to the dataset being too small is decreasing the number of hyperparameters
    
Detecting Overfitting:
  We need to see the validation loss to determine overfitting. Higher validation loss means it's likely overfitting
  
  
