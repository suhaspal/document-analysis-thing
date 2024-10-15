import os
import csv
import torch
import transformers
import numpy as np
import pandas as pd
import torch.utils.data as Data
from transformers import DistilBertTokenizer, WEIGHTS_NAME, CONFIG_NAME, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

            self.name_list, self.sentence_list = [], []
            for sentence, _ in rows:
                begin = sentence.find('<e1>')
                end = sentence.find('</e1>')
                e1 = sentence[begin:end + 5]

                begin = sentence.find('<e2>')
                end = sentence.find('</e2>')
                e2 = sentence[begin:end + 5]

                self.name_list.append(e1 + " " + e2)
                self.sentence_list.append(sentence)
            print("Data processing complete.")

class DataConverter:
    def __init__(self, names, sentences, target, device):
        print("Initializing DataConverter...")
        self.input_ids, self.attention_mask = [], []
        for i in tqdm(range(len(sentences)), desc="Encoding Sentences"):
            encoded_dict = tokenizer.encode_plus(
                sentences[i],
                add_special_tokens=True,
                max_length=96,
                padding='max_length',
                truncation=True,  # Add this to truncate sequences longer than max_length
                return_tensors='pt',
            )
            self.input_ids.append(encoded_dict['input_ids'].to(device))
            self.attention_mask.append(encoded_dict['attention_mask'].to(device))

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)
        self.target = torch.LongTensor(target).to(device)
        print("Data encoding complete. Tensor shapes:")
        print(f"input_ids: {self.input_ids.shape}, attention_mask: {self.attention_mask.shape}")


class ModelEvaluator:
    def __init__(self, device):
        print("Initializing ModelEvaluator...")
        self.device = device
        self.best_score = 0

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def save(self, model):
        print("Saving the model with current best score...")
        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)
        print("Model saved.")

    def eval(self, model, validation_dataloader):
        print("Starting evaluation...")
        model.eval()
        eval_accuracy, nb_eval_steps = 0, 0
        for batch in tqdm(validation_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits = model(batch[0], attention_mask=batch[1])[0]
                logits = logits.detach().cpu().numpy()
                label_ids = batch[2].cpu().numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
        avg_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Accuracy: {avg_accuracy}")
        if self.best_score < avg_accuracy:
            self.best_score = avg_accuracy
            print("New best score achieved!")
            self.save(model)
        else:
            print("No improvement in score.")

transformers.logging.set_verbosity_error()
print("Loading DistilBERT tokenizer and setting device...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

output_dir = './models/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

filename = 'train.csv'
data_processor = DataProcessor(filename)
name_list = data_processor.name_list
sentence_list = data_processor.sentence_list
label_list = data_processor.label_list
classes_list = data_processor.classes_list
num_classes = data_processor.num_classes

data_converter = DataConverter(name_list, sentence_list, label_list, device)
input_ids = data_converter.input_ids
attention_mask = data_converter.attention_mask
labels = data_converter.target

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created the '{output_dir}' directory.")

print("Splitting data into training and validation sets...")
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1, test_size=0.1)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.1)
print("Data split complete.")

train_data = Data.TensorDataset(train_inputs, train_mask, train_labels)
train_dataloader = Data.DataLoader(train_data, batch_size=32, shuffle=True)
validation_data = Data.TensorDataset(val_inputs, val_mask, val_labels)
validation_dataloader = Data.DataLoader(validation_data, batch_size=32, shuffle=True)
print("Data loaders created.")

print("Initializing DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=num_classes).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)
print("Optimizer configured.")

epoch = 2
evaluator = ModelEvaluator(device)
print("Starting training loop...")
for epoch_num in tqdm(range(epoch), desc="Training Epochs"):
    print(f"Epoch {epoch_num + 1}/{epoch}")
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
        batch = tuple(t.to(device) for t in batch)
        model.train()
        loss = model(batch[0], attention_mask=batch[1], labels=batch[2])[0]
        print(f"Batch {i + 1}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Evaluating at batch {i + 1}...")
            evaluator.eval(model, validation_dataloader)

print("Training complete. Reloading tokenizer and model for prediction...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification.from_pretrained(output_dir)
model.eval()
model.to(device)

def predict_relation(text):
    print(f"Predicting relation for input: {text}")
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=96,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()
    
    predicted_label = classes_list[predicted_class]
    print(f"Predicted relation: {predicted_label}")
    return predicted_label

# Test prediction
text = "<e1>Microsoft</e1> likes to manufacture <e2>computers</e2>."
predicted_relation = predict_relation(text)
print(f"Final Predicted relation: {predicted_relation}")
