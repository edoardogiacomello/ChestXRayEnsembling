# Chest X-Ray Ensembling
Respository for the paper on Chest X-Ray Ensembling (WIP).

## Files and folders:
### Original Code
 - Tesi: Sources of the original Thesis from Luca Nassano
 - Results: 
       - *.csv: Results that have been included in the paper. Filenames that includes "tta", "nomeans", etc are from later experiments that are not included in the paper.
 - Chexpert_Classification.ipynb: Original code from Nassano
 - CAM.ipynb: Original notebook from Nassano
 - Check_Unconditional_Dataset.ipynb: Conditional label analysis of the dataset.
 - Covid.ipynb: Code from Nassano for Covid data
 - chexpert_write_tfrecords.py: Code from Nassano for generating dataset (incomplete)
 
### Later Experiments (not included in Thesis or Paper)
 - Results: 
    - v2: Results for a new round of training
    - Models, ConditionalTraining: Results from other experiments, please refer to corresponding notebooks.
 - Chexpert_Classification-WithTTA.ipynb: An implementation of Test-Time-Augmentation
 - Chexpert_Classification-v2.ipynb: Another ensembling configuration, not included in the paper
 - Chexpert_Experiments.ipynb: Latest experiments tried on ensembling techniques, not included
 - Dev*.ipynb: Test notebooks used during development
 - HieraricalTraining.py: Implementation of hierarchical loss (instead of the one used in our paper)
 - model_conditional_pretraining.py: Script version for pre-training the networks
 - model_finetuning.py: Script version for fine-tuning the networks
 - save_tta_predictions_and_embeddings.py: Script for generating predictions/embeddings using tta for later analysis
 - *.csv: Temporary results
 
