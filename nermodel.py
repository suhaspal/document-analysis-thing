import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AdamW
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification 
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

device = torch.device("cuda")
print(f"Using device: {device}")

dataset = load_dataset("conll2003", trust_remote_code=True)
label_list = dataset["train"].features["ner_tags"].feature.names
label_to_id = {label: id for id, label in enumerate(label_list)}
id_to_label = {id: label for label, id in label_to_id.items()}
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

def tokenize_and_align_labels(examples):
    tokenize_input = tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=256, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenize_input.word_ids(batch_index=i)
        previous_word = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word:
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)    
            else:
                label_ids.append(-100)
            previous_word = word_idx
        labels.append(label_ids)
    tokenize_input["labels"] = labels
    return tokenize_input

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-cased", num_labels=len(label_list))
model.to(device)
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./results_ner",
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    no_cuda=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

model = DistilBertForTokenClassification.from_pretrained("./ner_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./ner_model")

device = torch.device("cuda")
model.to(device)


def predict_ner(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)   
    
    predictions = torch.argmax(outputs.logits, dim=2)
    
    predicted_labels = [id_to_label[pred.item()] for pred in predictions[0]]
    
    aligned_labels = []
    for word, label in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predicted_labels):
        if word.startswith("##"):
            continue
        if word in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        aligned_labels.append((word, label))
    
    return aligned_labels

text = "John Smith works at Microsoft in New York."
results = predict_ner(text)

print("Named Entities:")
for word, label in results:
    if label != "O":
        print(f"{word}: {label}")