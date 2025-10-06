# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project implements a Random Forest Classifier using the scikit-learn framework. 
This model was developed as part of the Udacity Deploying a Scalable ML Pipeline with FastAPI project to predict whether an individual earns more than $50,000 per year or less than or equal to $50,000 based on demographic and occupational data from the UCI Adult Census Income dataset.
The artifacts produced by the pipeline include a trained model (model.pkl) and an encoder (encoder.pkl) used for categorical feature transformation. The model is trained and evaluated entirely in Python 3.10 within a controlled Conda environment.

## Intended Use
The model is intended for educational and demonstration purposes only. 
It serves as a practical learning example of how to build, evaluate, and deploy a machine learning model using FastAPI and continuous integration workflows. 
The primary goal is to illustrate the concepts of reproducible ML pipelines, model evaluation, and slice-based performance reporting.
This model should not be used for production systems, hiring decisions, or financial assessments, as it has not been audited for fairness, bias, or robustness in these use cses.

## Training Data
The model was trained on the UCI Adult Census Income dataset, which contains approximately 40,000 samples of U.S. census records. 
The data includes both continuous and categorical features such as age, education, workclass, marital-status, occupation, relationship, race, sex, hours-per-week, and native-country.
Categorical variables were one-hot encoded using scikit-learn’s OneHotEncoder, and the target label (salary) was binarized into two classes: >50K and <=50K. 
The data was split into 80% for training and 20% for testing to evaluate model performance on unseen examples.

## Evaluation Data
The evaluation dataset consists of the held 20% of the data, which was not used during training. 
It mirrors the feature distribution of the training set and provides an unbiased estimate of model performance.
Performance was also assessed on demographic data slices (e.g., by education, occupation, sex, and race) to examine how predictions vary across subgroups.

## Metrics
The model was evaluated using Precision, Recall, and F1 Score, which measure correctness, sensitivity, and balance between precision and recall, respectively.
Metric	Description	Test Score
Precision	Proportion of positive predictions that were correct	0.7327
Recall	Proportion of actual positives correctly identified	0.6397
F1 Score	Harmonic mean of precision and recall	0.6830

The model performs reasonably well in distinguishing income classes but shows moderate recall, indicating it misses some higher-income cases.

Performance varies across subgroups, as shown in the slice analysis (slice_output.txt).
For example:
By workclass, F1 scores ranged from 0.50 (“?”) to 0.77 (“Self-emp-inc”).
By education, scores ranged from 0.25 (“9th”) to 0.89 (“Doctorate”).
By sex, both male and female groups achieved similar F1 values near 0.68.
By native-country, results varied widely due to small sample sizes; the U.S. group achieved an F1 of 0.68, while low-count countries often produced perfect or zero values because of limited data.

These variations suggest that the model’s reliability decreases in smaller or underrepresented subgroups.

## Ethical Considerations
This model is trained on census data that reflects historical biases and socioeconomic disparities.
As a result, its predictions may carry embedded biases related to race, gender, nationality, or occupation.
Although the slice performance evaluation helps identify uneven performance, this project does not include bias mitigation or fairness adjustments.

## Caveats and Recommendations
The model was trained with default Random Forest parameters and without hyperparameter tuning.
Small demographic groups lead to unreliable performance metrics in some slices, especially for rare occupations or countries.
Future improvements could include hyperparameter optimization, cross-validation, and bias mitigation analysis.
For real-world applications, additional steps such as fairness audits, interpretability, and model calibration would be critical for success.