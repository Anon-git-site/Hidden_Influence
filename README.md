# Hidden_Influence
This repository supports the paper, "The Hidden Influence of Linearity & Magnitude on Supervised Classification."

For a list of requirements, please see requirements.txt. 

## Base training and evaluation code & data

The base training and evaluation code for CIFAR-10 is located in cifar_train.py. This file can be modified to train and evaluate INaturalist and Places. For these datasets, we used 40 training epochs instead of 200 (CIFAR-10); and a Resnet-56 vs. a Resnet-32 architectures. The Resnet modules are contained in resnet_cifar.py. 

The INaturalist data can be found at https://github.com/visipedia/inat_comp/tree/master/2017 and is subject to a separate terms of use and license: https://www.inaturalist.org/pages/terms. The Places dataset is located at http://places2.csail.mit.edu/ and is subject to a Creative Common License (Attribution CC BY) license. For CIFAR-10, the imbalanced training and validation files are generated in cifar_train.py and for Places and INaturalist, they are located in the data folder. The CIFAR-10 data can be automatically downloaded via Torchvision. 

The source code for REMIX is available at https://github.com/cbellinger27/remix; and EOS at https://github.com/dd1github/EOS.

Pre-trained image models are located in the models folder.  There are 3 versions of each method because results were averaged over 3 cuts of the data.

Implementations of SMOTE and ADASYN can be found at: https://github.com/analyticalmindsltd/smote_variants.

## Relatationship of Frequency and Magnitude 
The following folders support the relationship of frequency and magnitude of latent features discussion in the paper.
The frequency folder contains csv files for 8 datasets with classification embedding (CE) frequencies. The csv file rows correspond to classes and the columns correspond to the frequency with which a top-K feature index appears in all instances in a class. The CE columns are sorted based on the CE index position that occurs most frequently in a class. Hence, column 0 corresponds to the frequency of the CE index that occurs most often in the set of top-K CE that are required to predict an individual instance of a class.

The magnitude folder contains csv files for 8 datasets with classification embedding (CE) magnitudes. The csv file rows correspond to classes and the columns correspond to CE values arranged by vector index position. The CE columns are sorted based on the corresonding frequency index. Hence, column 0 corresponds to the magnitude of the element in the CE vector with the highest frequency.
