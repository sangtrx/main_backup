import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from model import OCR
from data import OCRDataset
from torch.utils.data.dataset import random_split


vocab = '~0123456789-no'
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[2] for item in batch])
    text = [torch.tensor([stoi[c] for c in item[1]]) for item in batch]
    max_length = max([len(t) for t in text])
    texts = torch.zeros(len(text), max_length)
    for i, text_ in enumerate(text):
        texts[i, :len(text_)] = text_
    return images, texts, lengths


def prepare_dataloaders(batch_size, train_ratio=0.7):
    image_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor()
    ])
    dataset = OCRDataset(image_transform)
    # train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    # val_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    train_lengths = int(train_ratio * len(dataset))
    val_lengths = len(dataset) - train_lengths
    train_dataset, val_dataset = random_split(dataset, [train_lengths, val_lengths], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader


def save_checkpoint(path):
    torch.save({
        'vocab': vocab,
        'model': model.state_dict()
    }, path)


def train_1_epoch(model, criterion, optimizer, train_loader, val_loader):
    global device
    global best_acc
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        images, targets, lengths = batch
        images = images.to(device)
        targets = targets.to(device)
        log_probs = model(images, targets)  # T, B, V
        input_lengths = torch.full((log_probs.size(1),), log_probs.size(0))  # B
        loss = criterion(log_probs, targets, input_lengths, lengths)
        print(f'Loss {batch_idx+1:5d} / {len(train_loader)}', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_samples = 0
    correct_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, targets, lengths = batch
            images = images.to(device)
            probs = model(images)  # B, T, V
            predicts = probs.argmax(-1)  # B, T
            num_samples += len(images)
            for pred, tgt in zip(predicts.cpu(), targets):
                pred = ''.join([itos[x.item()] for x in pred])
                pred = pred[0] + ''.join([c for i, c in enumerate(pred[1:], 1) if c != pred[i-1]])
                pred = pred.replace('~', '')
                tgt = ''.join([itos[x.item()] for x in tgt])
                tgt = tgt.replace('~', '')
                print(f'"{pred}" "{tgt}"')
                correct_samples += 1 if pred == tgt else 0
    accuracy = correct_samples / num_samples
    if accuracy > best_acc:
        best_acc = accuracy
        save_checkpoint('best.pth')
    print(f'ACC = {best_acc:.02f}')

torch.manual_seed(0)
device = 'cuda'
model = OCR(len(vocab))
model.to(device)
criterion = nn.CTCLoss(blank=stoi['~'])
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9, verbose=True)
train_loader, val_loader = prepare_dataloaders(batch_size=8, train_ratio=0.7)
max_epoch = 1000
best_acc = 0.0

for epoch in range(max_epoch):
    print(f'Epoch {epoch}')
    train_1_epoch(model, criterion, optimizer, train_loader, val_loader)
    lr_scheduler.step()
    save_checkpoint('latest.pth')