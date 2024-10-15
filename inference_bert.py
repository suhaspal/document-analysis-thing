import torch
import csv
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Load the tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Load test data
def load_test_data(filename):
    sentences, labels = [], []
    with open(filename) as f:
        rows = [row for row in csv.reader(f)]
        rows = np.array(rows[1:])  # Exclude header
        label_list = [label for _, label in rows]
        classes_list = list(set(label_list))  # Unique labels for classes
        
        for sentence, label in rows:
            sentences.append(sentence)
            labels.append(classes_list.index(label))  # Convert labels to class indices
    return sentences, labels, classes_list

# Process data for inference
def process_data(sentences):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence, add_special_tokens=True, max_length=96, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
        token_type_ids.append(encoded_dict["token_type_ids"])
    
    if input_ids and attention_masks and token_type_ids:
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(token_type_ids, dim=0)
    else:
        raise ValueError("The input data for sentences is empty, please check the 'test.csv' file.")

# Evaluate the model
# Evaluate the model
def evaluate_model(model, input_ids, attention_masks, token_type_ids, labels):
    with torch.no_grad():
        logits = model(input_ids.to(device), attention_mask=attention_masks.to(device), token_type_ids=token_type_ids.to(device))[0]
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Convert to numpy array
        labels = labels.cpu().numpy()  # Convert labels to numpy array
        accuracy = np.sum(predictions == labels) / len(labels)  # Calculate accuracy
    return accuracy


# Load test set
a,b,classes_list = load_test_data("train.csv")
test_sentences, test_labels, c = load_test_data("test.csv")
test_input_ids, test_attention_masks, test_token_type_ids = process_data(test_sentences)
test_labels = torch.tensor(test_labels)

# Initialize and load model
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(classes_list))
model.load_state_dict(torch.load("bert_model_weights.pth", map_location=device))
model.to(device)
model.eval()

# Evaluate and print accuracy
accuracy = evaluate_model(model, test_input_ids, test_attention_masks, test_token_type_ids, test_labels)
print(f"Test Accuracy: {accuracy:.4f}")
