import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the combined movie reviews (not labeled one)
df = pd.read_csv('C:/Users/Raed Karkoub/Desktop/DS320/combined_movie_reviews.csv')

# Prepare features and labels
X = df['review_text']
y = df['source']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Predict and evaluate Logistic Regression
lr_preds = lr_model.predict(X_test_tfidf)
print("\nLogistic Regression Results (Source Prediction):")
print(classification_report(y_test, lr_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))

# Predict and evaluate Random Forest
rf_preds = rf_model.predict(X_test_tfidf)
print("\nRandom Forest Results (Source Prediction):")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))

# ==========================
# Feature Importance for Logistic Regression
# ==========================

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get coefficients for each class
coef = lr_model.coef_[0]

# Get top 20 features for IMDB (positive coefficients) and RottenTomatoes (negative coefficients)
top_positive_indices = np.argsort(coef)[-20:]
top_negative_indices = np.argsort(coef)[:20]

# Plot top features
plt.figure(figsize=(12, 6))
plt.barh(range(20), coef[top_positive_indices], align='center', color='green')
plt.yticks(range(20), feature_names[top_positive_indices])
plt.title('Top 20 words associated with IMDB')
plt.xlabel('Coefficient Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('top_words_imdb.png')  # Save the plot
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(range(20), coef[top_negative_indices], align='center', color='red')
plt.yticks(range(20), feature_names[top_negative_indices])
plt.title('Top 20 words associated with RottenTomatoes')
plt.xlabel('Coefficient Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('top_words_rt.png')  # Save the plot
plt.show()

print("Feature importance plots saved: top_words_imdb.png and top_words_rt.png")