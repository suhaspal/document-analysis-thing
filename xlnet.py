import os
import csv
import torch
import transformers
import numpy as np
import pandas as pd
import torch.utils.data as Data

from transformers import AutoTokenizer,WEIGHTS_NAME,CONFIG_NAME,XLNetForSequenceClassification,AdamW
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, filename):
        with open(filename) as f:
            rows = [row for row in csv.reader(f)]
            rows = np.array(rows[1:])
            self.label_list = [label for _, label in rows] 
            self.classes_list = list(set(self.label_list))
            self.num_classes = len(self.classes_list)
            for i in range(len(self.label_list)):
                self.label_list[i] = self.classes_list.index(self.label_list[i]) 

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

class DataConverter:
    def __init__(self, names, sentences, target):
        self.input_ids, self.token_type_ids, self.attention_mask = [], [], []
        for i in range(len(sentences)):
            encoded_dict = tokenizer.encode_plus(
                sentences[i],  
                add_special_tokens = True,   
                max_length = 96,      
                pad_to_max_length = True,
                return_tensors = 'pt',       
            )
            self.input_ids.append(encoded_dict['input_ids'])
            self.token_type_ids.append(encoded_dict['token_type_ids'])
            self.attention_mask.append(encoded_dict['attention_mask'])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.token_type_ids = torch.cat(self.token_type_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)

        self.input_ids = torch.LongTensor(self.input_ids)
        self.token_type_ids = torch.LongTensor(self.token_type_ids)
        self.attention_mask = torch.LongTensor(self.attention_mask)
        self.target = torch.LongTensor(target)

class ModelEvaluator:
    def __init__(self, device):
        self.device = device
        self.best_score = 0

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def save(self, model):
        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)

    def eval(self, model, validation_dataloader):
        model.eval()
        eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
        for batch in validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
                logits = logits.detach().cpu().numpy()
                label_ids = batch[3].cpu().numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        if self.best_score < eval_accuracy / nb_eval_steps:
            self.best_score = eval_accuracy / nb_eval_steps
            self.save(model)

transformers.logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_score = 0
batch_size = 32
classes_list = list()

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

data_converter = DataConverter(name_list, sentence_list, label_list)
input_ids = data_converter.input_ids
token_type_ids = data_converter.token_type_ids
attention_mask = data_converter.attention_mask
labels = data_converter.target

import os

if not os.path.exists('./models'):
    os.makedirs('./models')
    print("Created the './models' directory.")

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1, test_size=0.1)
train_token, val_token, _, _ = train_test_split(token_type_ids, labels, random_state=1, test_size=0.1)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.1)

train_data = Data.TensorDataset(train_inputs, train_token, train_mask, train_labels)
train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

validation_data = Data.TensorDataset(val_inputs, val_token, val_mask, val_labels)
validation_dataloader = Data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_classes).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

epoch = 2
for _ in range(epoch):
    for i, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        evaluator = ModelEvaluator(device)
        if i % 1 == 0:
            evaluator.eval(model, validation_dataloader)

from transformers import XLNetForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

model = XLNetForSequenceClassification.from_pretrained('./models')
model.eval() 
model.to(device) 

def predict_relation(text):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=96,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    token_type_ids = encoded_dict['token_type_ids'].to(device)
    

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()
    
    predicted_label = classes_list[predicted_class]
    
    return predicted_label

text = "<e1>Microsoft</e1> likes to manufacture <e2>computers</e2>."
predicted_relation = predict_relation(text)
print(f"Predicted relation: {predicted_relation}")