# EENG645A_FinalProject
Final Project repository for Practical Machine Learning with Major Machin

### GitHub Organization
This repository was set up on AceHub and run using the __git.antcenter.net:4567/nyielding/acehub-tensorflow-image:latest__ image with 32 Gi of memory and 1 GPU. 

There are several important files to pay attention to here. All data preprocessing is done in the preprocess.py file. Simply uncomment all the code and ensure that the all the files you want to preprocess are correctly downloaded and stored in the raw_data file. All the data can be found from the following website: https://globalfishingwatch.org/data-download/datasets/public-training-data-v1. Not all the data could be stored on GitHub, however, as many of the files are too large for GitHub to accept. If you run the code on your own, please ensure you have the following .csv files downloaded and stored in the raw_data folder: fixed_gear.csv, pole_and_line.csv, purse_seines.csv, trawlers.csv, trollers.csv, and unknown.csv. During the preprocessing process, new dataframes will be stored in the processed_data folder. Because these file sizes are too large, this folder was otherwise empty. If you run into issues when running the code *make sure to make a local version of the __processed_data__ folder in your own repository!*

The next important file is the ann.py file. This function holds the artificial nueral network (ANN) that is used to create the initial model, overfit model, generalized model, and the hyperparameter tuned model. While preprocessing has been completed, this file also does the data splitting for the training, testing, and validation sets and saves this to the split_data folder. Storing the split data in the raw_data folder was the only way to logically break up the dataset in such a way that it would be small enough to upload to GitHub, as we are dealing with hundreds of thousands of observations. To train a model (or multiple in the same run) simply uncomment the block of code beneath the respective header. As a warning, the hyperparameter tuned model takes over 10 hours to fully run!

Finally, a separate script was created to create confusion matrices. All figures created during the running of any script will be stored in the figures folder, and all models will be stored in the models folder. Happy grading!

### Project Deliverables
The project deliverables are a PDF report, a PowerPoint presentation, and this GitHub repository. The PDF report is titled __Final Project Report.pdf__ and the PowerPoint presentation is titled __Final Project Presentation.pptx__. Please contact me ASAP if any of the files or deliverables are not accessible, and I will get them to you in another format immediately.
