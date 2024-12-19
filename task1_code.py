#1. Setup and Imports
!pip install transformers datasets torch sklearn
!pip install datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset, Dataset
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/trainset/orientation/orientation-tr-train.tsv'
data = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')
#2. Printing the Data Set
print(data.head())

#3. Tokenization and Dataset Preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_and_prepare(data, text_column):
    texts = data[text_column].fillna("").tolist()  # Handle missing values
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {
        'input_ids': encodings['input_ids'].detach().numpy(),  
        'attention_mask': encodings['attention_mask'].detach().numpy(),
        'labels': data['label'].to_numpy()  
    }

encoded_data_en = tokenize_and_prepare(data, 'text_en')
encoded_data_orig = tokenize_and_prepare(data, 'text')

train_data, test_data = train_test_split(data, test_size=0.1, stratify=data['label'])

def create_dataset(encoded_data, indices):
    return Dataset.from_dict({
        'input_ids': np.array([encoded_data['input_ids'][i] for i in indices]),
        'attention_mask': np.array([encoded_data['attention_mask'][i] for i in indices]),
        'labels': np.array([encoded_data['labels'][i] for i in indices])
    })

train_dataset_en = create_dataset(encoded_data_en, train_data.index)
test_dataset_en = create_dataset(encoded_data_en, test_data.index)
train_dataset_orig = create_dataset(encoded_data_orig, train_data.index)
test_dataset_orig = create_dataset(encoded_data_orig, test_data.index)


#4. Model Training and Evaluation Function

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    report_to='none'  # Disable external logging
)

model_en = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
trainer_en = Trainer(
    model=model_en,
    args=training_args,
    train_dataset=train_dataset_en,
    eval_dataset=test_dataset_en,
    compute_metrics=compute_metrics
)
trainer_en.train()

model_orig = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
trainer_orig = Trainer(
    model=model_orig,
    args=training_args,
    train_dataset=train_dataset_orig,
    eval_dataset=test_dataset_orig,
    compute_metrics=compute_metrics
)
trainer_orig.train()


#5. Zero-Shot Inference and Comparative Analysis
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sample_texts_en = test_data['text_en'].dropna().sample(2).tolist()
sample_texts_orig = test_data['text'].dropna().sample(2).tolist()

predictions_en = zero_shot_classifier(sample_texts_en, candidate_labels=["left", "right"])
predictions_orig = zero_shot_classifier(sample_texts_orig, candidate_labels=["left", "right"])
print("Zero-shot classification results for English text:", predictions_en)
print("Zero-shot classification results for original language text:", predictions_orig)
eval_results_en = trainer_en.evaluate()
eval_results_orig = trainer_orig.evaluate()

print("Evaluation results for English model:", eval_results_en)
print("Evaluation results for Original Language model:", eval_results_orig)


