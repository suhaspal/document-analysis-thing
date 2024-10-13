import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AdamW
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification 
from transformers import Trainer, TrainingArguments


device = torch.device("cuda")
print(f"Using device: {device}")


dataset = load_dataset("conll2003", trust_remote_code=True)
label_list = dataset["train"].features["ner_tags"].feature.names
label_to_id = {label: id for id, label in enumerate(label_list)}
id_to_label = {id: label for label, id in label_to_id.items()}
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize_and_align_labels(examples):
    tokenize_input = tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenize_input.word_ids(batch_index=1)
        previous_word = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word:
                if word_idx<len(label):
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
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))
model.to(device)
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir= "./results",
    evaluation_strategy= 'epoch',
    learning_rate= 2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    no_cuda=False
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator

)

trainer.train()

model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")