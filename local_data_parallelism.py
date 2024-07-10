import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import torch.distributed as dist


class CustomTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_bbox_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, epochs=3):
    print(f"Starting training on rank {rank}.")
    setup(rank, world_size)

    # Load data
    df = pd.read_csv('Datasets/dataset_final.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    incorrect = [str(r) for r in df['Incorrect']]
    correct = [str(r) for r in df['Correct']]
    
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

    dataset = CustomTextDataset(encodings)

    # DDP: Distribute the dataset across the GPUs using DistributedSampler.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    num_labels = len(unique_correct)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    print("Model loaded...")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)

        for batch_num, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if batch_num % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch+1}, Loss: {loss.item()}")

    cleanup()
    print(f"Training complete on rank {rank}.")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)