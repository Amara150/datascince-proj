# datascince-proj
Task1:
# Exploratory Data Analysis on Titanic Dataset

This project performs **Exploratory Data Analysis (EDA)** on the Titanic passenger dataset to uncover patterns in survival outcomes. It includes data cleaning, visualizations, and interpretation of key features like passenger class, gender, fare, and embarkation port.

---

# Dataset

Source: [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data) or [Data Science Dojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
Size: 891 records, 12 features
Key Features:
   `Survived`: 0 = No, 1 = Yes
   `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
   `Sex`, `Age`, `Fare`, `Embarked`, etc.

##  Project Steps
# 1. Data Loading
Load dataset using pandas.
Display first few rows and dataset structure using `.head()` and `.info()`.
# 2. Data Cleaning
Handle missing values:
   Fill `Age` with median (numerical).
   Fill `Embarked` with mode (categorical).
   Drop `Cabin` due to too many missing values.
Remove duplicate records.
Identify and understand outliers using boxplots (e.g., `Fare` and `Age`).
# 3. Data Visualization
Bar plots: for categorical features: `Sex`, `Pclass`, `Embarked`, `Survived`
Histograms: for numeric features: `Age`, `Fare`, `SibSp`, `Parch`
Correlation heatmap: for numeric features (after encoding `Sex`):
   `Sex_encoded = 0` (male), `1` (female)
# 4. Insights and Interpretation
 Females and 1st-class passengers had higher survival rates.
 Most passengers in 3rd class did not survive.
 Fare and survival have a weak positive correlation (~0.26).
 Passenger class and survival are moderately negatively correlated (~-0.34).
 Positive correlation between being female and surviving (~0.54).
# How to Run This Project
# Prerequisites
Python 3.x
 Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`
# Steps
1. Open the Jupyter Notebook or Colab file: `Titanic_EDA.ipynb`
2. Run each cell step-by-step.
3. Visualizations will appear below each cell.
4. Adjust or explore new features as needed.
 ** Sample Outputs**
 # Correlation Heatmap
 `Pclass vs Survived`: -0.34
 `Fare vs Survived`: 0.26
 `Sex_encoded vs Survived`: 0.54
# Boxplot Observations
Fare has many valid high-value outliers (1st-class passengers).
Age is right-skewed; majority are between 20–40.
# Observations

Socioeconomic status influenced survival chances.
Gender was a strong predictor: females had much higher survival rates.
Embarkation port had minor influence, but Southampton (S) had the most passengers.
# Future Improvements
Perform feature engineering (e.g., `FamilySize`, `Title` from names).
 Build predictive models using classification algorithms.
 Apply clustering for passenger segmentation.

Task2:
# Sentiment Analysis on IMDB Movie Reviews 

This project performs **binary sentiment classification** on the IMDB Movie Reviews Dataset using **Natural Language Processing (NLP)** and **Machine Learning**. It predicts whether a movie review expresses a **positive** or **negative** sentiment.

# Dataset
Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  Size: 50,000 movie reviews
  Features:
`review`: Full text of the movie review
`sentiment`: Label (`positive` or `negative`)

# Project Steps

# 1. Data Loading
Load the IMDB dataset CSV using pandas.
 Convert sentiment labels to binary (positive → 1, negative → 0).

# 2. Text Preprocessing
 Lowercase conversion
 Tokenization (split sentences into words)
 Remove stopwords (e.g., "the", "is", "and")
 Remove punctuation
 Lemmatization (e.g., "running" → "run")

# 3. Feature Engineering
- Use TF-IDF vectorization to convert text into numerical features.

# 4. Model Training
Train a Logistic Regression classifier on 80% of the dataset.

# 5. Model Evaluation
 Evaluate using Accuracy, Precision, Recall, and F1-score.
 Achieved accuracy: ~88.8%

# 6. Predict New Sentiment
Custom function accepts a user input review and returns:
   `"Positive "` or `"Negative "`

# How to Run This Project

# Prerequisites
Python 3.x
Libraries: `nltk`, `pandas`, `scikit-learn`

# Setup (Google Colab or Local)
1. Clone or upload the repository to your environment.
2. Download the IMDB dataset from Kaggle.
3. Adjust the dataset file path in the script.
4. Run the script cells step-by-step in `IMDB_Sentiment_Analysis.ipynb`.

# Sample Output

# Credit Card Fraud Detection System

This project builds a machine learning system to detect fraudulent credit card transactions using the popular Credit Card Fraud Detection dataset. It includes data preprocessing (with SMOTE to handle class imbalance), model training using Random Forest, performance evaluation, and a simple command-line testing interface.

Task 3

# Dataset

Source: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  Records 284,807 transactions
  Features: 
   `V1` to `V28`: PCA-transformed features
   `Amount`: Transaction amount
   `Class`: 1 = Fraud, 0 = Legitimate

# Project Workflow

# 1. Data Preprocessing

 Dropped `Time` column
 Scaled `Amount` feature using `StandardScaler` → stored as `Amount_scaled`
 Handled severe class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)

#2. Model Training

 Split data into 80% training and 20% test
 Trained a **Random Forest Classifier** on the SMOTE-balanced training set

# 3. Model Evaluation

Evaluated using:
  Accuracy
  Precision
   Recall
   F1-score
Achieved strong recall on fraud class (important to catch frauds while minimizing false negatives)

# 4. Testing Interface

Implemented a manual input interface:
Accepts 29 values (V1–V28 and transaction `Amount`)
Automatically scales the `Amount`
Predicts: "FRAUD DETECTED!"or "Transaction is legitimate."


