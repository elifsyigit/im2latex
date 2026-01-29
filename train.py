import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import BATCH_SIZE, LEARNING_RATE, DEVICE, PAD_ID
from data.dataset import Im2LatexDataset
from models.transformer import Im2LatexModel


def train():
    print("Loading dataset...")
    train_dataset = Im2LatexDataset(split='train')
    val_dataset = Im2LatexDataset(split='val',tokenizer=train_dataset.tokenizer)
    
    vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Vocab size: {vocab_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("Initializing model...")
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "im2latex_colab.pt"
    best_checkpoint_path = checkpoint_dir / "im2latex_colab_best.pt"

    model = Im2LatexModel(vocab_size=vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    start_epoch = 0
    
    print("Starting training from scratch...")
    
    model.train()
    num_epochs = 20
    print_interval = 100
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {DEVICE}")
    print(f"Print loss every {print_interval} steps\n")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = 0.0           # windowed loss for logging
        window_steps = 0             # steps in current logging window
        epoch_loss_sum = 0.0         # sum of loss over entire epoch
        epoch_steps = 0              # number of steps in epoch
        epoch_correct_tokens = 0     # correct non-PAD tokens in epoch
        epoch_total_tokens = 0       # total non-PAD tokens in epoch
        
        # ---------- Training loop ----------
        for step, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_tokens = batch['input_tokens'].to(DEVICE ,non_blocking=True)
            target_tokens = batch['target_tokens'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(images, input_tokens)
            
            logits = logits.view(-1, vocab_size)
            targets = target_tokens.view(-1)

            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            running_loss += batch_loss
            window_steps += 1
            epoch_loss_sum += batch_loss
            epoch_steps += 1

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                non_pad_mask = (targets != PAD_ID)
                correct = (preds == targets) & non_pad_mask
                epoch_correct_tokens += correct.sum().item()
                epoch_total_tokens += non_pad_mask.sum().item()
            
            if (step + 1) % print_interval == 0:
                avg_loss = running_loss / max(window_steps, 1)
                print(f"Epoch {epoch + 1}, Step {step + 1}, "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0
                window_steps = 0
        
        # ---------- Compute training epoch metrics ----------
        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_token_acc = (epoch_correct_tokens / epoch_total_tokens) if epoch_total_tokens > 0 else 0.0

        # ---------- Validation loop ----------
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_correct_tokens = 0
        val_total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE, non_blocking=True)
                input_tokens = batch['input_tokens'].to(DEVICE)
                target_tokens = batch['target_tokens'].to(DEVICE)

                logits = model(images, input_tokens)
                logits = logits.view(-1, vocab_size)
                targets = target_tokens.view(-1)

                loss = criterion(logits, targets)
                val_loss_sum += loss.item()
                val_steps += 1

                preds = logits.argmax(dim=-1)
                non_pad_mask = (targets != PAD_ID)
                correct = (preds == targets) & non_pad_mask
                val_correct_tokens += correct.sum().item()
                val_total_tokens += non_pad_mask.sum().item()

        val_avg_loss = val_loss_sum / max(val_steps, 1)
        val_token_acc = (val_correct_tokens / val_total_tokens) if val_total_tokens > 0 else 0.0

        # ---------- Checkpointing ----------
        is_best = val_avg_loss < best_val_loss
        if is_best:
            best_val_loss = val_avg_loss

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "vocab_size": vocab_size,
            "tokenizer": train_dataset.tokenizer,
        }

        # Save "last" checkpoint every epoch
        torch.save(checkpoint_state, last_checkpoint_path)

        # Save "best" checkpoint only when validation improves
        if is_best:
            torch.save(checkpoint_state, best_checkpoint_path)

        # Switch back to training mode for next epoch
        model.train()

        # ---------- Log epoch summary ----------
        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} completed.")
        print(f"  Train  - loss: {epoch_avg_loss:.4f}, token accuracy: {epoch_token_acc:.4f}")
        print(f"  Val    - loss: {val_avg_loss:.4f}, token accuracy: {val_token_acc:.4f}")
        if is_best:
            print(f"  Checkpoint: new best model saved to {best_checkpoint_path}")
        print(f"  Checkpoint: last model saved to {last_checkpoint_path}\n")


if __name__ == "__main__":
    train()

