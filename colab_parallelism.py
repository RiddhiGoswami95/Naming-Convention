import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


# Custom Dataset class
class CustomTextTokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
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
encodings = tokenizer(incorrect, correct, truncation=True, padding='max_length', max_length=50, return_tensors='pt')

# Generate the unique class labels and index them
unique_correct = list(set(correct))
labels_map = {name: idx for idx, name in enumerate(unique_correct)}
labels = torch.tensor([labels_map[name] for name in correct])
encodings['labels'] = labels

print(f"Label Distribution: {torch.unique(labels, return_counts=True)}")

# Create a dataset and a data loader with the custom dataset
dataset = CustomTextTokenizedDataset(encodings)
loader = DataLoader(dataset, batch_size=16)

# Define the model with the correct number of classes
num_labels = len(unique_correct)
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
print("Model loaded...")

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Detect if we have more than one GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    # This line is the key to using multiple GPUs
    model = torch.nn.DataParallel(model)

# Move our model to the appropriate device
model.to(device)

# Training loop
print("Starting Training...")

for epoch in range(3):
    model.train()
    print(f"Starting epoch {epoch+1}...")
    for batch_num, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Batch {batch_num} complete!')

    print(f'Epoch {epoch+1} complete!')

# Example end of training, evaluate your model performance as required
print("Training complete!")