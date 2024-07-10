import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


# Custom Dataset class
class CustomTextDataset(Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


# Helper function to check data loading works
def check_dataloader(loader):
    print("Checking data loader...")
    for batch_num, batch in enumerate(loader):
        print(f"Batch {batch_num} loaded!")
        if batch_num == 0:
            print(batch)


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

# Check if the DataLoader works properly
# check_dataloader(loader)

# Define the model with the correct number of classes
num_labels = len(unique_correct)
model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                      num_labels=num_labels)
print("Model loaded...")

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Starting Training...")

for epoch in range(3):
    model.train()
    print(f"Starting epoch {epoch+1}...")
    for batch_num, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Batch {batch_num} complete!')

    print(f'Epoch {epoch+1} complete!')

    # Evaluate the model
    model.eval()
    correct_count = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()

    accuracy = correct_count / total
    print(f'Accuracy after epoch {epoch+1}: {accuracy:.4f}')

print("Training complete!\n")
