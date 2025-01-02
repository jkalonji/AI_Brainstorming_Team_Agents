

import sentence_transformers
import numpy as np

# Split the knowledge base into chunks
chunks = knowledge_base.split('\n\n')

# Load the pre-trained model
model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# Function to get the most similar chunk
def get_most_similar_chunk(query, chunks, chunk_embeddings, model):
    query_embedding = model.encode([query])
    # Compute cosine similarities
    similarities = np.dot(chunk_embeddings, query_embedding.T)
    # Find the index of the most similar chunk
    most_similar_index = np.argmax(similarities)
    return chunks[most_similar_index]



if __name__ == "__main__":
    query = "How to load the Boston Housing dataset ?"
    answer = get_most_similar_chunk(query, chunks, chunk_embeddings, model)
    print(answer)


    knowledge_base = """
# Knowledge Base for Machine Learning Exercises

This document serves as a comprehensive knowledge base for students working on the provided machine learning exercises. Each exercise focuses on different regression techniques using the Boston Housing dataset. The document covers the theoretical background, implementation steps, and practical considerations for each exercise.

## Table of Contents

1. **Introduction to Regression**
2. **Simple Linear Regression**
3. **Multiple Linear Regression**
4. **Polynomial Regression**
5. **Ridge and Lasso Regression**
6. **Advanced Regression Techniques**
7. **Evaluation Metrics**
8. **Tools and Libraries**
9. **Common Issues and Troubleshooting**

---

## 1. Introduction to Regression

### What is Regression?
Regression analysis is a statistical method used to examine the relationship between a dependent variable (target) and one or more independent variables (features). It is commonly used for prediction and forecasting.

### Types of Regression
- **Simple Linear Regression:** Involves one dependent variable and one independent variable.
- **Multiple Linear Regression:** Involves one dependent variable and multiple independent variables.
- **Polynomial Regression:** Captures non-linear relationships by adding polynomial features.
- **Ridge and Lasso Regression:** Regularized regression techniques to prevent overfitting.
- **Advanced Regression Techniques:** Includes Support Vector Regression, Decision Tree Regression, and Random Forest Regression.

---

## 2. Simple Linear Regression

### Overview
Simple Linear Regression models the relationship between a dependent variable and a single independent variable.

### Steps to Implement
1. **Load the Dataset:**
   - Use `sklearn.datasets` to load the Boston Housing dataset.
2. **Select the Feature:**
   - Choose the 'LSTAT' feature, which represents the percentage of lower status of the population.
3. **Split the Data:**
   - Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.
4. **Train the Model:**
   - Use `LinearRegression` from `sklearn.linear_model` to train the model on the training data.
5. **Make Predictions:**
   - Predict housing prices on the test set.
6. **Evaluate the Model:**
   - Calculate the Mean Squared Error (MSE) using `mean_squared_error` from `sklearn.metrics`.
7. **Visualize Results:**
   - Plot the actual vs. predicted values using `matplotlib`.

### Key Concepts
- **Mean Squared Error (MSE):** A measure of the quality of an estimator. Lower values indicate better fit.
- **Scatter Plot:** Visualizes the relationship between two variables.

---

## 3. Multiple Linear Regression

### Overview
Multiple Linear Regression extends simple linear regression by including multiple independent variables.

### Steps to Implement
1. **Load the Dataset:**
   - Use `sklearn.datasets` to load the Boston Housing dataset.
2. **Select Features:**
   - Choose features such as 'LSTAT', 'RM' (average number of rooms per dwelling), and 'AGE' (proportion of owner-occupied units built prior to 1940).
3. **Feature Scaling:**
   - Standardize features using `StandardScaler` from `sklearn.preprocessing` if necessary.
4. **Split the Data:**
   - Split the dataset into training and testing sets.
5. **Train the Model:**
   - Use `LinearRegression` to train the model.
6. **Evaluate the Model:**
   - Calculate MSE and R² score using `mean_squared_error` and `r2_score` from `sklearn.metrics`.
7. **Interpret Coefficients:**
   - Analyze the impact of each feature on the target variable.

### Key Concepts
- **R² Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Feature Impact:** The influence of each feature on the target variable, determined by the coefficients of the model.

---

## 4. Polynomial Regression

### Overview
Polynomial Regression captures non-linear relationships by transforming features into polynomial features of a specified degree.

### Steps to Implement
1. **Select a Feature:**
   - Choose a feature suspected to have a non-linear relationship with the target variable.
2. **Create Polynomial Features:**
   - Use `PolynomialFeatures` from `sklearn.preprocessing` to create polynomial features of degree 2 and 3.
3. **Train the Model:**
   - Use `LinearRegression` to train the polynomial regression model.
4. **Evaluate the Model:**
   - Compare MSE and R² score with the linear model.
5. **Visualize the Fit:**
   - Plot the regression curves to visualize how well the model fits the data.

### Key Concepts
- **Polynomial Features:** Transformed features that allow linear models to fit non-linear relationships.
- **Overfitting:** When a model is too complex and captures noise in the data.

---

## 5. Ridge and Lasso Regression

### Overview
Ridge and Lasso Regression are regularization techniques that help prevent overfitting by adding a penalty to the loss function.

### Steps to Implement
1. **Load the Dataset:**
   - Use the Boston Housing dataset with multiple features.
2. **Standardize Features:**
   - Use `StandardScaler` to standardize the features.
3. **Train Ridge and Lasso Models:**
   - Use `Ridge` and `Lasso` from `sklearn.linear_model` with different alpha values.
4. **Evaluate Using Cross-Validation:**
   - Use cross-validation to evaluate model performance.
5. **Compare Coefficients:**
   - Analyze the coefficients to understand the impact of regularization.
6. **Choose Best Alpha:**
   - Use grid search to find the optimal alpha value.

### Key Concepts
- **Regularization:** Techniques to reduce the complexity of a model to prevent overfitting.
- **Alpha Value:** The regularization parameter that controls the strength of the penalty.
- **Cross-Validation:** A method to assess how well a model generalizes to an independent dataset.
- **Grid Search:** A method to tune hyperparameters by searching through a specified parameter grid.

---

## 6. Advanced Regression Techniques

### Overview
Advanced regression techniques include Support Vector Regression (SVR), Decision Tree Regression, and Random Forest Regression, which offer more flexibility and power compared to linear models.

### Steps to Implement
1. **Choose an Algorithm:**
   - Select one of SVR, Decision Tree Regression, or Random Forest Regression.
2. **Apply the Algorithm:**
   - Use the chosen algorithm to train a model on the Boston Housing dataset.
3. **Tune Hyperparameters:**
   - Use grid search or random search to tune hyperparameters.
4. **Evaluate Performance:**
   - Compare the performance with linear and polynomial models using appropriate metrics.
5. **Discuss Advantages and Disadvantages:**
   - Consider aspects like model interpretability, complexity, and performance.

### Key Concepts
- **Support Vector Regression (SVR):** Uses the same principles as Support Vector Machines for classification.
- **Decision Tree Regression:** A tree-based model that splits the data into subsets based on feature values.
- **Random Forest Regression:** An ensemble method that fits multiple decision trees and averages the results.
- **Hyperparameter Tuning:** The process of selecting the best set of hyperparameters for a model.
- **Model Interpretability:** How easily the model's predictions can be explained.

---

## 7. Evaluation Metrics

### Common Metrics
- **Mean Squared Error (MSE):** The average of the squares of the errors.
- **R² Score:** The coefficient of determination, indicating how well the model explains the variability of the response data.
- **Cross-Validation Score:** Provides a measure of how well the model generalizes to an independent dataset.

---

## 8. Tools and Libraries

### Key Libraries
- **scikit-learn:** For loading datasets, preprocessing, model selection, and evaluation.
- **matplotlib:** For data visualization.
- **numpy:** For numerical operations.

### Functions to Use
- `sklearn.datasets.load_boston()`: To load the Boston Housing dataset.
- `sklearn.model_selection.train_test_split()`: To split the dataset.
- `sklearn.preprocessing.StandardScaler()`: For feature scaling.
- `sklearn.preprocessing.PolynomialFeatures()`: To create polynomial features.
- `sklearn.linear_model.LinearRegression()`, `Ridge()`, `Lasso()`: For regression models.
- `sklearn.svm.SVR()`: For Support Vector Regression.
- `sklearn.tree.DecisionTreeRegressor()`: For Decision Tree Regression.
- `sklearn.ensemble.RandomForestRegressor()`: For Random Forest Regression.
- `sklearn.metrics.mean_squared_error()`, `r2_score()`: For evaluation metrics.
- `sklearn.model_selection.GridSearchCV()`: For hyperparameter tuning.

---

## 9. Common Issues and Troubleshooting

### Issues to Watch Out For
- **Overfitting:** Ensure that the model generalizes well to unseen data.
- **Feature Scaling:** Some models are sensitive to the scale of features.
- **Polynomial Degree:** Higher degrees can lead to overfitting.
- **Alpha Value Selection:** Choose an appropriate alpha value for regularization.

### Tips for Troubleshooting
- **Cross-Validation:** Use cross-validation to assess model performance.
- **Grid Search:** Systematically search for the best hyperparameters.
- **Visualization:** Use plots to understand the relationships and model fit.

---

## Conclusion

This knowledge base provides a comprehensive guide to understanding and implementing various regression techniques using the Boston Housing dataset. By following the steps and understanding the concepts, students can effectively complete the exercises and gain a deeper insight into regression analysis in machine learning.
"""