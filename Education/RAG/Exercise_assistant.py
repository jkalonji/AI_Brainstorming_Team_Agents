from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

exercises= {
  "exercises": [
    {
      "id": "R1E1",
      "title": "Simple Linear Regression",
      "description": "Implement a simple linear regression model using a single feature from the Boston housing dataset.",
      "difficulty": "easy",
      "algorithm": ["Simple Linear Regression"],
      "dataset": "Boston Housing Dataset",
      "steps": [
        "Load the Boston housing dataset using scikit-learn.",
        "Select the 'LSTAT' feature as the independent variable.",
        "Split the dataset into training and testing sets.",
        "Train a simple linear regression model on the training data.",
        "Predict housing prices on the test set and calculate the mean squared error (MSE).",
        "Plot the actual vs. predicted values on a scatter plot."
      ],
      "prerequisites": [],
      "tags": ["simple linear regression", "single feature", "LSTAT", "MSE", "scatter plot"],
      "answer": ""
    },
    {
      "id": "R1E2",
      "title": "Multiple Linear Regression",
      "description": "Extend the previous exercise to include multiple features and build a multiple linear regression model.",
      "difficulty": "medium",
      "algorithm": ["Multiple Linear Regression"],
      "dataset": "Boston Housing Dataset",
      "steps": [
        "Use the same Boston housing dataset.",
        "Select multiple features such as 'LSTAT', 'RM', and 'AGE'.",
        "Perform feature scaling if necessary.",
        "Split the data into training and testing sets.",
        "Train a multiple linear regression model.",
        "Evaluate the model using MSE and R² score.",
        "Discuss the impact of each feature on the housing prices."
      ],
      "prerequisites": ["R1E1"],
      "tags": ["multiple linear regression", "feature scaling", "MSE", "R² score", "feature impact"],
      "answer": ""
    },
    {
      "id": "R1E3",
      "title": "Polynomial Regression",
      "description": "Apply polynomial regression to capture non-linear relationships in the data.",
      "difficulty": "medium",
      "algorithm": ["Polynomial Regression"],
      "dataset": "Boston Housing Dataset",
      "steps": [
        "Choose a feature from the Boston housing dataset that you suspect has a non-linear relationship with the target variable.",
        "Create polynomial features of degree 2 and 3.",
        "Train a polynomial regression model.",
        "Compare the performance of the polynomial model with the linear model using MSE and R².",
        "Plot the regression curves to visualize the fit."
      ],
      "prerequisites": ["R1E1"],
      "tags": ["polynomial regression", "non-linear relationship", "polynomial features", "MSE", "R²", "regression curves"],
      "answer": ""
    },
    {
      "id": "R1E4",
      "title": "Ridge and Lasso Regression",
      "description": "Implement Ridge and Lasso regression to understand the effect of regularization.",
      "difficulty": "hard",
      "algorithm": ["Ridge Regression", "Lasso Regression"],
      "dataset": "Boston Housing Dataset",
      "steps": [
        "Use the Boston housing dataset with multiple features.",
        "Standardize the features.",
        "Train Ridge and Lasso regression models with different alpha values.",
        "Evaluate the models using cross-validation.",
        "Compare the coefficients of the models and discuss the impact of regularization.",
        "Choose the best alpha value using grid search."
      ],
      "prerequisites": ["R1E2", "R1E3"],
      "tags": ["ridge regression", "lasso regression", "regularization", "alpha values", "cross-validation", "grid search"],
      "answer": ""
    },
    {
      "id": "R1E5",
      "title": "Advanced Regression Techniques",
      "description": "Explore advanced regression techniques and model selection.",
      "difficulty": "hard",
      "algorithm": ["Support Vector Regression", "Decision Tree Regression", "Random Forest Regression"],
      "dataset": "Boston Housing Dataset",
      "steps": [
        "Select one of the following algorithms: Support Vector Regression (SVR), Decision Tree Regression, or Random Forest Regression.",
        "Apply the chosen algorithm to the Boston housing dataset.",
        "Tune the hyperparameters using grid search or random search.",
        "Compare the performance of the advanced model with the linear and polynomial models.",
        "Discuss the advantages and disadvantages of the chosen algorithm in the context of this dataset.",
        "Consider the model's interpretability and complexity in your discussion."
      ],
      "prerequisites": ["R1E4"],
      "tags": ["advanced regression", "SVR", "decision trees", "random forests", "hyperparameter tuning", "model comparison", "interpretability"],
      "answer": ""
    }
  ]
}

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([ex['description'] for ex in exercises])

# Function to retrieve top k exercises based on query
def retrieve_exercises(query, top_k=3):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]
    return [exercises[i] for i in top_indices]

# Function to rank exercises by difficulty
def rank_exercises(retrieved_exercises):
    return sorted(retrieved_exercises, key=lambda x: x['difficulty'])

# Function to generate a response based on retrieved exercises
def generate_response(retrieved_exercises):
    response = "Based on your query, here are the relevant exercises:\n"
    for ex in retrieved_exercises:
        response += f"- {ex['id']} - {ex['description']} (Difficulty: {ex['difficulty']})\n"
    return response

# Main function to run the RAG system
def rag_system(query, top_k=3):
    retrieved = retrieve_exercises(query, top_k)
    ranked = rank_exercises(retrieved)
    response = generate_response(ranked)
    print(response)

# Example usage
if __name__ == "__main__":
    query = "I want to practice classification algorithms."
    rag_system(query, top_k=3)