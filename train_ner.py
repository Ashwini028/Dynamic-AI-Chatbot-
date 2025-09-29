import os
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# ---------------------------
# 1. Load BIO Data
# ---------------------------
def load_bio(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        words, tags = [], []
        for line in f:
            if line.strip() == "":
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, tag = parts
                else:
                    word, tag = parts[0], "O"   # fallback
                words.append(word)
                tags.append(tag)
        if words:
            sentences.append(words)
            labels.append(tags)
    return sentences, labels


train_sentences, train_labels = load_bio(r"D:\AI Chatbot Project\data\engtrain.bio")
test_sentences, test_labels = load_bio(r"D:\AI Chatbot Project\data\engtest.bio")

# ---------------------------
# 2. Build tag set from BOTH train & test
# ---------------------------
unique_tags = sorted(set(tag for doc in (train_labels + test_labels) for tag in doc))

# Ensure "O" exists
if "O" not in unique_tags:
    unique_tags.append("O")

tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

print("Unique tags:", unique_tags)
print("num_tags =", len(unique_tags))

# ---------------------------
# 3. Tokenizer
# ---------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_and_align(sentences, labels):
    encodings = tokenizer(sentences, is_split_into_words=True, truncation=True, padding=True)
    new_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(tag2id.get(label[word_id], tag2id["O"]))  # safe fallback
            else:
                label_ids.append(tag2id.get(label[word_id], tag2id["O"]))
            prev_word_id = word_id
        new_labels.append(label_ids)
    encodings["labels"] = new_labels
    return encodings


train_encodings = tokenize_and_align(train_sentences, train_labels)
test_encodings = tokenize_and_align(test_sentences, test_labels)

train_dataset = Dataset.from_dict(train_encodings)
test_dataset = Dataset.from_dict(test_encodings)

# ---------------------------
# 4. Model
# ---------------------------
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(unique_tags),
    id2label=id2tag,
    label2id=tag2id
)

# ---------------------------
# 5. Training
# ---------------------------
training_args = TrainingArguments(
    output_dir="models/ner_model",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    logging_dir="logs/ner_logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# ---------------------------
# 6. Save model
# ---------------------------
os.makedirs("models/ner_model", exist_ok=True)
model.save_pretrained("models/ner_model")
tokenizer.save_pretrained("models/ner_model")

print(" NER model saved in models/ner_model/")
