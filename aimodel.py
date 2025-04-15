import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re

# 1. Load your dataset
print("Loading dataset...")
df = pd.read_csv("all_data.csv")

# 2. Data cleaning
print("Cleaning data...")
# Drop rows with missing values in comment_text
df = df.dropna(subset=['comment_text'])
# Fill missing toxicity values with 0
df['toxicity'] = df['toxicity'].fillna(0)

# 3. Filter for 'toxicity' binary classification
df['label'] = (df['toxicity'] > 0.5).astype(int)  # Toxic if > 0.5

# 4. Clean text
print("Preprocessing text...")
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['comment_clean'] = df['comment_text'].apply(clean_text)
# Remove empty strings after cleaning
df = df[df['comment_clean'].str.len() > 0]

# 5. Vectorization with both word and character n-grams
print("Creating features...")
# Word-level TF-IDF with n-grams (1-3)
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    max_features=5000
)

# Character-level TF-IDF with n-grams (3-5)
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=5000
)

# Fit and transform both vectorizers
X_word = word_vectorizer.fit_transform(df['comment_clean'])
X_char = char_vectorizer.fit_transform(df['comment_clean'])

# Combine word and character features
from scipy.sparse import hstack
X = hstack([X_word, X_char])
y = df['label']

# 6. Train/test split (80/20)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train logistic regression with L2 regularization
print("Training model...")
model = LogisticRegression(
    penalty='l2',  # L2 regularization
    C=1.0,         # Inverse of regularization strength
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# 8. Predict
print("Making predictions...")
y_pred = model.predict(X_test)

# 9. Calculate metrics
print("\nModel Performance Metrics:")
print("=" * 50)

# Calculate individual metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 50)

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Save
print("\nSaving model and vectorizers...")
joblib.dump(model, "logistic_model.pkl")
joblib.dump(word_vectorizer, "word_vectorizer.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")
print("Done!")

def predict_toxicity(comment: str, model, word_vectorizer, char_vectorizer) -> str:
    # Clean text
    comment_clean = clean_text(comment)
    
    # Vectorize with both vectorizers
    X_word = word_vectorizer.transform([comment_clean])
    X_char = char_vectorizer.transform([comment_clean])
    X = hstack([X_word, X_char])
    
    # Predict
    prediction = model.predict(X)[0]
    
    return "Toxic" if prediction == 1 else "Non-Toxic"
