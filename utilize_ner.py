from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch


model = DistilBertForTokenClassification.from_pretrained("./ner_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./ner_model")


device = torch.device("cuda")
model.to(device)


model.eval()

id_to_label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

def predict_ner(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)


    predictions = torch.argmax(outputs.logits, dim=2)

    predicted_labels = [id_to_label[label_id.item()] for label_id in predictions[0]]


    words = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_labels = []
    current_word = ""
    current_label = ""
    for word, label in zip(words, predicted_labels):
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_word:
                word_labels.append((current_word, current_label))
            current_word = word
            current_label = label
    if current_word:
        word_labels.append((current_word, current_label))

    return word_labels

text = "John Smith works at Microsoft in New York."
results = predict_ner(text)
print(results)