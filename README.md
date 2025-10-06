# rainfall-prediction

🌧️ Rainfall Prediction using Machine Learning

📌 Project Overview

This project aims to predict rainfall using various weather parameters through Machine Learning algorithms in Python.
By analyzing historical weather data, the model predicts whether it will rain on a given day — helping in agriculture planning, water resource management, and disaster preparedness.

🧠 Objective

To develop a Rainfall Prediction Model that accurately forecasts rainfall using supervised learning techniques based on meteorological features such as temperature, humidity, pressure, and wind speed.

⚙️ Approach

Data Collection:

The dataset used consists of daily weather observations (temperature, humidity, wind speed, pressure, etc.).

Data was sourced from publicly available weather datasets such as Kaggle or government meteorological data portals.

Data Cleaning & Preprocessing:

Handled missing values using imputation techniques (mean/median/mode).

Converted categorical data (like wind direction or rainfall status) into numerical form using Label Encoding / One-Hot Encoding.

Scaled numerical features for better model performance using StandardScaler or MinMaxScaler.

Exploratory Data Analysis (EDA):

Visualized key weather patterns using matplotlib and seaborn.

Checked feature correlations and their influence on rainfall occurrence.

Identified trends such as “High humidity and low temperature increase rainfall probability.”

Model Building:

Split the dataset into training (80%) and testing (20%) sets.

Tested multiple algorithms:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

XGBoost

Evaluated each model using metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Model Evaluation & Optimization:

Compared model performances and selected the one with the best trade-off between precision and recall.

Applied hyperparameter tuning using GridSearchCV to optimize results.

Prediction & Visualization:

Created a function to predict rainfall given new weather input values.

Visualized actual vs predicted results and confusion matrix to understand prediction accuracy.

📊 Key Findings

Humidity and temperature were the most significant predictors of rainfall.

Random Forest outperformed other models with the highest accuracy (e.g., ~88%).

Areas with higher average humidity (>80%) showed a strong likelihood of rainfall.

Feature scaling and balanced datasets significantly improved performance.

🛠️ Technologies Used

Programming Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Tools: Jupyter Notebook / VS Code

🏁 Conclusion

This project demonstrates how machine learning can be effectively applied to predict rainfall, providing valuable insights for farmers, meteorologists, and planners. Through data-driven methods, we can better understand weather dependencies and make informed decisions.
