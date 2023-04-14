# Hidden_Influence
This repository supports the paper, "The Hidden Influence of Linearity & Magnitude on Supervised Classification."

For a list of requirements, please see requirements.txt.

The base training and evaluation code for CIFAR-10 is located in XX. 

This file can be modified to train and evaluate INaturalist and Places. FOr these datasets, we used 40 training epochs instead of 200 (CIFAR-10); and a Resnet-56 vs. a Resnet-32 architectures. The Resnet modules are located at XX. 

The INaturalist data can be found at https://paperswithcode.com/dataset/inaturalist and Places at http://places2.csail.mit.edu/. The imbalanced training and validation files are located at 

The source code for REMIX is available at https://github.com/cbellinger27/remix; and EOS at https://github.com/dd1github/EOS.

Pre-trained models are located in the folder XX.  

Implementations of SMOTE and ADASYN can be found at: https://github.com/analyticalmindsltd/smote_variants.
