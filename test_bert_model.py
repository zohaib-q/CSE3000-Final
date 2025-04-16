import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
MAX_LENGTH = 128

def predict_toxicity(text, model, tokenizer, device):
    # Tokenize input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Toxic" if prediction == 1 else "Non-Toxic"

def main():
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = BertForSequenceClassification.from_pretrained("bert_toxicity_model")
        tokenizer = BertTokenizer.from_pretrained("bert_toxicity_model")
        model = model.to(device)
        print("Model loaded successfully!")

        # Interactive testing
        print("\nEnter comments to test (type 'exit' to quit):")
        while True:
            comment = input("\nEnter a comment: ")
            if comment.lower() == 'exit':
                break

            result = predict_toxicity(comment, model, tokenizer, device)
            print(f"Prediction: {result}")

    except FileNotFoundError:
        print("Error: Model files not found. Please run bert_model.py first to train the model.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 