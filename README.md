# Hidden_Influence
This repository supports the paper, "The Hidden Influence of Linearity & Magnitude on Supervised Classification."

For a list of requirements, please see requirements.txt. 

## Base training and evaluation code & data

### Image data

The base training and evaluation code for CIFAR-10 is located in cifar_train.py. This file can be modified to train and evaluate INaturalist and Places. For these datasets, we used 40 training epochs instead of 200 (CIFAR-10); and a Resnet-56 vs. a Resnet-32 architectures. The Resnet modules are contained in resnet_cifar.py. 

The INaturalist data can be found at https://github.com/visipedia/inat_comp/tree/master/2017 and is subject to a separate terms of use and license: https://www.inaturalist.org/pages/terms. The Places dataset is located at http://places2.csail.mit.edu/ and is subject to a Creative Common License (Attribution CC BY) license. For CIFAR-10, the imbalanced training and validation files are generated in cifar_train.py and for Places and INaturalist, they are located in the data folder. The CIFAR-10 data can be automatically downloaded via Torchvision. 

The source code for REMIX is available at https://github.com/cbellinger27/remix; and EOS at https://github.com/dd1github/EOS. DSM follows the same basic training and evaluation procedure as EOS, except that it incorporates SMOTE instead of nearest enemy class feature manipulation - see https://arxiv.org/pdf/2304.05895.pdf.

Pre-trained image models are located in the models folder under the CIFAR-10, Places and INaturalist sub-folders. There are 3 versions for each method because results were averaged over 3 cuts of the data.

The cifar_FE.py and resnet_cifar_FE.py modules provide mechanism for extracting feature embeddings (FE) from CNN architectures, which are, in turn, used to generate classification embeddings (CE).

### Tabular data

The logistic regression (LG) and support vector machine (SVM) training and evaluation models were drawn from the SKLEARN library using default settings: https://scikit-learn.org/stable/modules/svm.html and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.  The training and test data, used for 5-way cross-validation, for these models can be found at: https://drive.google.com/file/d/1M9xgweB3IcPh1k2WlRYhJtURsc5YOtDg/view?usp=share_link.

Implementations of SMOTE and ADASYN can be found at: https://github.com/analyticalmindsltd/smote_variants.

## RQ1: Number of Features (CE) Required to Predict an Instance
This section provides additional details regarding the methodology for computing the classification embeddings (CE), the percentage of CE as a ratio of the model classification layer dimension, and the average magnitude of CE. The module cif_CE_mag_all.py illustrates how CE are extracted from pre-trained models for CIFAR-10. This method can be adapted for Places and INaturalist. All pre-trained models can be found in the models folder. The output from cif_CE_mag_all.py should be stored in a folder (here, it is named "XX"). The output of cif_CE_mag_all.py is a .csv file that contains the following information for each instance in a dataset: target class, predicted class, class instance index, length of CE (i.e., the number of CE required to predict an instance), the index of the CE in the classification embedding layer dimension, and the magnitude of the CE. Sample output is contained in the folder XX. This output is then consumed by cif_CE_info.py to generate summary CE information for each dataset and class for purposes of the figures displayed in the paper.

## RQ3: Relatationship of Frequency and Magnitude 
The following folders support the relationship of frequency and magnitude of latent features discussion in the paper.
The frequency folder contains csv files for 8 datasets with classification embedding (CE) frequencies. The csv file rows correspond to classes and the columns correspond to the frequency with which a top-K feature index appears in all instances in a class. The CE columns are sorted based on the CE index position that occurs most frequently in a class. Hence, column 0 corresponds to the frequency of the CE index that occurs most often in the set of top-K CE that are required to predict an individual instance of a class.

The magnitude folder contains csv files for 8 datasets with classification embedding (CE) magnitudes. The csv file rows correspond to classes and the columns correspond to CE values arranged by vector index position. The CE columns are sorted based on the corresonding frequency index. Hence, column 0 corresponds to the magnitude of the element in the CE vector with the highest frequency.
