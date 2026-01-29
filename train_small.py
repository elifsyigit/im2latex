import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import BATCH_SIZE, LEARNING_RATE, DEVICE, PAD_ID
from data.dataset import Im2LatexDataset
from models.transformer import Im2LatexModel


def train():
    print("Loading dataset...")
    full_dataset = Im2LatexDataset(split='train')
    
    vocab_size = full_dataset.tokenizer.vocab_size
    print(f"Full dataset size: {len(full_dataset)}")
    
    small_dataset_size = 100
    indices = list(range(min(small_dataset_size, len(full_dataset))))
    train_dataset = Subset(full_dataset, indices)
    
    print(f"Using small dataset size: {len(train_dataset)}")
    print(f"Vocab size: {vocab_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print("Initializing model...")
    model = Im2LatexModel(vocab_size=vocab_size)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    num_epochs = 100
    print_interval = 10
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {DEVICE}")
    print(f"Print loss every {print_interval} steps\n")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_steps = 0
        
        for step, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_tokens = batch['input_tokens'].to(DEVICE)
            target_tokens = batch['target_tokens'].to(DEVICE)
            
            optimizer.zero_grad()
            
            logits = model(images, input_tokens)
            
            logits = logits.view(-1, vocab_size)
            targets = target_tokens.view(-1)
            
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_steps += 1
            
            if (step + 1) % print_interval == 0:
                avg_loss = running_loss / total_steps
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
                total_steps = 0
        
        epoch_avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {epoch_avg_loss:.4f}\n")
    
    print("Saving model...")
    model_path = Path("checkpoints")
    model_path.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'tokenizer': full_dataset.tokenizer
    }, model_path / "im2latex_model_small.pt")
    print(f"Model saved to {model_path / 'im2latex_model_small.pt'}")


if __name__ == "__main__":
    train()

