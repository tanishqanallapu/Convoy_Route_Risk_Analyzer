from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------
dataset = load_dataset('csv', data_files={'data': 'data/terrorism_text_dataset.csv'})['data']
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ---------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
#model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# ---------------------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/terrorism/",
    eval_strategy="epoch",     # ✅ evaluate after each epoch
    num_train_epochs=2,
    per_device_train_batch_size=2,
    report_to="none"
)

# ---------------------------------------------------------------------
# METRIC FUNCTION (only accuracy)
# ---------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ---------------------------------------------------------------------
# TRAINER
# ---------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------
# TRAIN AND EVALUATE
# ---------------------------------------------------------------------
trainer.train()

results = trainer.evaluate()
print(f"\n✅ Model Accuracy: {results['eval_accuracy'] * 100:.2f}%")

# ---------------------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------------------
model.save_pretrained("models/terrorism/")
tokenizer.save_pretrained("models/terrorism/")
print("✅ Model trained and saved to models/terrorism/") 