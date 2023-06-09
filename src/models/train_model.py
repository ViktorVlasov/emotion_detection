import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


df_train_full = pd.read_csv('rusentitweet_train_v2.csv')
df_test = pd.read_csv('rusentitweet_test_v2.csv')

le = preprocessing.LabelEncoder()
le.fit(df_train_full['label'])
df_train_full['label'] = le.transform(df_train_full['label'])
df_test['label'] = le.transform(df_test['label'])

df_train, df_validation = train_test_split(df_train_full, 
                                           test_size=0.15, 
                                           random_state=42)

df_validation[['text', 'label']].to_csv('rusentitweet_validation_hf.csv', index=False)
df_train[['text', 'label']].to_csv('rusentitweet_train_hf.csv', index=False)
df_test[['text', 'label']].to_csv('rusentitweet_test_hf.csv', index=False)

dataset = load_dataset('csv', data_files={
    'train': 'rusentitweet_train_hf.csv', 
    'test': 'rusentitweet_test_hf.csv', 
    'validation': 'rusentitweet_validation_hf.csv'
    })

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def tokenize_function(examples):
		return tokenizer(examples["text"], 
						padding="max_length",
						truncation=True, 
						max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, 
																	predictions, 
																	average='macro')
		acc = accuracy_score(labels, predictions)
		return {
				'accuracy': acc,
				'f1': f1,
				'precision': precision,
				'recall': recall
		}


EPOCHS = 4
BATHC_SIZE = 32

model_predictions = []

for seed_number in range(3):
	model = AutoModelForSequenceClassification.from_pretrained(
     "DeepPavlov/rubert-base-cased", 
     num_labels=5)
 
	training_args = TrainingArguments(
			output_dir="test_trainer", 
			evaluation_strategy="epoch",
			save_strategy="epoch",
			num_train_epochs = EPOCHS,
			overwrite_output_dir = 'True',
			per_device_train_batch_size=BATHC_SIZE,
			warmup_ratio = 0.1,
			learning_rate = 2e-5,
			seed=seed_number,
			metric_for_best_model='f1',
			load_best_model_at_end=True,
		)
 
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset = tokenized_datasets['train'],
		eval_dataset=tokenized_datasets['validation'],
		compute_metrics=compute_metrics
	)
 
	trainer.train()
	predictions = trainer.predict(tokenized_datasets["test"])
	preds = np.argmax(predictions.predictions, axis=-1)
	model_predictions.append(preds)
 
	print(classification_report(tokenized_datasets["test"]['label'], 
                             preds, 
                             digits=6, 
                             target_names=le.classes_))
	trainer.save_model('trainer_rubert_rusentitweet_seed'+str(seed_number))
	model.save_pretrained('rubert_rusentitweet_seed'+str(seed_number))