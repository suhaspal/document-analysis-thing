import os
import csv
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, WEIGHTS_NAME, CONFIG_NAME
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.utils.data as Data

class DataProcessor:
    def __init__(self, filename):
        print("Initializing DataProcessor...")
        with open(filename) as f:
            rows = [row for row in csv.reader(f)]
            rows = np.array(rows[1:])
            self.label_list = [label for _, label in rows]
            self.classes_list = list(set(self.label_list))
            self.num_classes = len(self.classes_list)
            for i in range(len(self.label_list)):
                self.label_list[i] = self.classes_list.index(self.label_list[i])
            print(f"Data loaded with {len(self.label_list)} samples and {self.num_classes} unique classes.")

            self.sentence_list, self.target_text_list = [], []
            for sentence, label in rows:
                self.sentence_list.append(f"Extract relation: {sentence}")
                self.target_text_list.append(label)
            print("Data processing complete.")

class DataConverter:
    def __init__(self, sentences, target_texts, device):
        print("Initializing DataConverter...")
        self.input_ids, self.attention_mask, self.labels = [], [], []
        for i in tqdm(range(len(sentences)), desc="Encoding Sentences"):
            input_encoded = tokenizer(sentences[i], max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            target_encoded = tokenizer(target_texts[i], max_length=10, padding='max_length', truncation=True, return_tensors="pt")

            self.input_ids.append(input_encoded.input_ids.to(device))
            self.attention_mask.append(input_encoded.attention_mask.to(device))
            self.labels.append(target_encoded.input_ids.to(device))

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        print("Data encoding complete. Tensor shapes:")
        print(f"input_ids: {self.input_ids.shape}, attention_mask: {self.attention_mask.shape}, labels: {self.labels.shape}")

class ModelEvaluator:
    def __init__(self, device):
        print("Initializing ModelEvaluator...")
        self.device = device
        self.best_score = float('inf')  # Initialize best_score for loss-based improvement

    def save(self, model):
        print("Saving the model with current best score...")
        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)
        print("Model saved.")

    def eval(self, model, validation_dataloader):
        print("Starting evaluation at end of epoch...")
        model.eval()
        eval_loss, correct_predictions, nb_eval_steps = 0, 0, 0
        total_samples = 0
        for batch in tqdm(validation_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
                loss = outputs.loss
                eval_loss += loss.item()

                # Generate predictions
                predictions = model.generate(input_ids=batch[0], attention_mask=batch[1], max_length=10)
                
                # Decode predictions and labels
                pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                label_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in batch[2]]
                
                # Calculate accuracy
                correct_predictions += sum(1 for pred, label in zip(pred_texts, label_texts) if pred.strip() == label.strip())
                total_samples += len(label_texts)
                nb_eval_steps += 1

        avg_loss = eval_loss / nb_eval_steps
        accuracy = correct_predictions / total_samples
        print(f"Validation Loss: {avg_loss}, Validation Accuracy: {accuracy * 100:.2f}%")

        if avg_loss < self.best_score:
            self.best_score = avg_loss
            print("New best score achieved!")
            self.save(model)
        else:
            print("No improvement in score.")

# Load the tokenizer and set the device
print("Loading T5 tokenizer and setting device...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

output_dir = './models/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# Prepare data
filename = 'train.csv'
data_processor = DataProcessor(filename)
sentence_list = data_processor.sentence_list
target_text_list = data_processor.target_text_list

# Convert data to tensors
data_converter = DataConverter(sentence_list, target_text_list, device)
input_ids = data_converter.input_ids
attention_mask = data_converter.attention_mask
labels = data_converter.labels

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created the '{output_dir}' directory.")

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1, test_size=0.1)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.1)
print("Data split complete.")

# Create DataLoader objects
train_data = Data.TensorDataset(train_inputs, train_mask, train_labels)
train_dataloader = Data.DataLoader(train_data, batch_size=16, shuffle=True)
validation_data = Data.TensorDataset(val_inputs, val_mask, val_labels)
validation_dataloader = Data.DataLoader(validation_data, batch_size=16, shuffle=True)
print("Data loaders created.")

# Initialize T5 model for conditional generation
print("Initializing T5 model...")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Configure optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
print("Optimizer configured.")

# Training and evaluation
epoch = 2
evaluator = ModelEvaluator(device)
print("Starting training loop...")
for epoch_num in tqdm(range(epoch), desc="Training Epochs"):
    print(f"Epoch {epoch_num + 1}/{epoch}")
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader, desc="Training Batches"):
        batch = tuple(t.to(device) for t in batch)
        outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch_num + 1} Training Loss: {epoch_loss / len(train_dataloader)}")

    # Evaluate at the end of each epoch
    evaluator.eval(model, validation_dataloader)

print("Training complete. Reloading tokenizer and model for prediction...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(output_dir)
model.eval()
model.to(device)

# Prediction function
def predict_relation(text):
    print(f"Predicting relation for input: {text}")
    input_text = f"Extract relation: {text}"
    encoded_dict = tokenizer.encode_plus(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Predicted relation: {prediction}")
    return prediction

# Test prediction
text = "<e1>Microsoft</e1> likes to manufacture <e2>computers</e2>."
predicted_relation = predict_relation(text)
print(f"Final Predicted relation: {predicted_relation}")
