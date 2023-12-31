# Explore-Multi-Label-Classification-with-an-Enzyme-Substrate-Dataset-Kaggle-

This project involved developing a machine learning model for multi-label classification of enzyme substrates, specifically EC1 and EC2 classes. The goal was to accurately predict the class of enzyme substrates from a given set of features. The project included several key steps:

**Data Preprocessing**: Before feeding the data into the model, it was essential to clean and preprocess it. This step involved handling missing values, normalizing or scaling the data, and encoding categorical variables. The dataset used in this project had 33 features, which required careful preprocessing to ensure that the model could learn from the data effectively.

**Model Selection**: The next step was to select an appropriate model for the task. XGBoost, a popular gradient boosting library, was chosen for this project because of its efficiency and performance in classification tasks.

**Hyperparameter Tuning**: After selecting the model, it was important to find the best set of hyperparameters to optimize its performance. RandomizedSearchCV was used to search over a grid of 16,250 different combinations of hyperparameters to find the best set. This method randomly samples a fixed number of hyperparameter combinations from a specified range and selects the combination that gives the best performance.

**Training and Prediction**: Once the optimal hyperparameters were identified, the XGBoost model was trained on the preprocessed dataset. After the model was trained, it was used to predict the probabilities of the enzyme substrate classes (EC1 and EC2) on a separate test dataset.

The model achieved an accuracy of 65.02%, which demonstrates its ability to classify enzyme substrates effectively. This project showcased skills in various aspects of machine learning, including data preprocessing, model selection, hyperparameter tuning, and making predictions using a gradient boosting library.
