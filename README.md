The goal of this project is to build a collaborative- filter based recommender system using the Million Song Dataset and the user interaction dataset from the Million Song Dataset Challenge. The collabora- tive filtering method assumes that people will listen to similar songs that they have listened to before and it makes recommendations based on the play count information on the songs. In this project, we used the collaborative filtering method with the Alternative Least Square model from PySpark’s built-in library to learn latent factor representations and conduct hyper- parameter tuning to improve the baseline and guard our model against overfitting. We further explored two extensions: Single Machine Implementation using the LightFM library and Fast Search for queries with the Annoy library to compare their performances with the baseline model.


## File Usage Description


### 1.baseline_full_script.py
This script includes all the codes for the baseline: train on downsampled data, tune hyper-parameters, make predictions on validation or test set, and evaluate using three metrics.


### 2.downsample_save.py
This script backs up downsampled train data after repartition.


### 3.model_save.py
This script saves the best-performing model and its userFactors and itemFactors.


### 4.lightfm_final.ipynb
This is a jupyter notebook that performs the LightFM extension on Greene.


### 5.fast_search.ipynb
This is a jupyter notebook that performs the Fast Search extension using Annoy locally.


### 6. Training25_Result.xlsx
This file shows all of our 25% training results.


### 7. 1004_Final_Project_Report.pdf
This is our final report.
