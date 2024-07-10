import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
import deepspeed


# Custom Dataset class
class CustomTextDataset(Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


# Load data
df2 = pd.read_csv('Datasets/dataset_final.csv')
print("Data loaded...")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Tokenizer loaded...")

incorrect = [str(r) for r in df2['Incorrect']]
correct = [str(r) for r in df2['Correct']]

# Preprocess the data
encodings = tokenizer(incorrect,
                      correct,
                      truncation=True,
                      padding='max_length',
                      max_length=50,
                      return_tensors='pt')

# Generate the unique class labels and index them
unique_correct = list(set(correct))
labels_map = {name: idx for idx, name in enumerate(unique_correct)}
labels = torch.tensor([labels_map[name] for name in correct])
encodings['labels'] = labels

print(f"Label Distribution: {torch.unique(labels, return_counts=True)}")

# Create a dataset and a data loader with the custom dataset
dataset = CustomTextDataset(encodings)
loader = DataLoader(dataset, batch_size=16)

# Define the model with the correct number of classes
num_labels = len(unique_correct)
model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                      num_labels=num_labels)
print("Model loaded...")

# Define DeepSpeed configuration
ds_config = {
    "train_batch_size": 16,  # Global batch size (adjust as per your setup)
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 1
    }
}

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
print("Optimizer loaded...")

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, model_parameters=model.parameters(), config_params=ds_config)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Starting Training...")

for epoch in range(3):
    model_engine.train()
    print(f"Starting epoch {epoch+1}...")
    for batch_num, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model_engine(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
        print(f'Batch {batch_num} complete!')

    print(f'Epoch {epoch+1} complete!')

    # Evaluate the model
    model_engine.eval()
    correct_count = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model_engine(input_ids=input_ids,
                                   attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()

    accuracy = correct_count / total
    print(f'Accuracy after epoch {epoch+1}: {accuracy:.4f}')

print("Training complete!\n")
