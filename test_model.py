import joblib
import sys
import re
from scipy.sparse import hstack

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

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

def main():
    try:
        # Load the saved model and vectorizers
        print("Loading model and vectorizers...")
        model = joblib.load("logistic_model.pkl")
        word_vectorizer = joblib.load("word_vectorizer.pkl")
        char_vectorizer = joblib.load("char_vectorizer.pkl")
        print("Model loaded successfully!")
        
        # Interactive testing
        print("\nEnter comments to test (type 'exit' to quit):")
        while True:
            comment = input("\nEnter a comment: ")
            if comment.lower() == 'exit':
                break
                
            result = predict_toxicity(comment, model, word_vectorizer, char_vectorizer)
            print(f"Prediction: {result}")
            
    except FileNotFoundError:
        print("Error: Model files not found. Please run aimodel.py first to train the model.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 