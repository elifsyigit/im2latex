import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))
from config import BATCH_SIZE, LEARNING_RATE, DEVICE, PAD_ID, SOS_ID, EOS_ID
from data.dataset import Im2LatexDataset
from models.transformer import Im2LatexModel
from evaluation.metrics import compute_token_accuracy, compute_exact_match


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
        epoch_start_time = time.time()
        
        # Reset GPU peak memory stats at start of epoch to measure training-only usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        
        running_loss = 0.0           # windowed loss for logging
        window_steps = 0             # steps in current logging window
        epoch_loss_sum = 0.0         # sum of loss over entire epoch
        epoch_steps = 0              # number of steps in epoch
        epoch_correct_tokens = 0     # correct non-PAD tokens in epoch
        epoch_total_tokens = 0       # total non-PAD tokens in epoch
        epoch_grad_norms = []        # gradient norms for each step
        
        # ---------- Training loop ----------
        for step, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_tokens = batch['input_tokens'].to(DEVICE ,non_blocking=True)
            target_tokens = batch['target_tokens'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(images, input_tokens)
            
            logits_flat = logits.view(-1, vocab_size)
            targets = target_tokens.view(-1)

            loss = criterion(logits_flat, targets)
            
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"Raw gradient norm: {raw_norm:.4f}, Clipped gradient norm: {clipped_norm:.4f}")                                                                          
            
            # Compute gradient norm (global L2) after backward, before optimizer.step() 
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            epoch_grad_norms.append(total_grad_norm)
            
            optimizer.step()
            
            batch_loss = loss.item()
            running_loss += batch_loss
            window_steps += 1
            epoch_loss_sum += batch_loss
            epoch_steps += 1

            with torch.no_grad():
                preds_flat = logits_flat.argmax(dim=-1)
                # Use metrics.py function for token accuracy
                batch_token_acc = compute_token_accuracy(preds_flat, targets, PAD_ID)
                # Accumulate for epoch average (approximate by counting tokens)
                non_pad_mask = (targets != PAD_ID)
                epoch_total_tokens += non_pad_mask.sum().item()
                epoch_correct_tokens += int(batch_token_acc * non_pad_mask.sum().item())
            
            if (step + 1) % print_interval == 0:
                avg_loss = running_loss / max(window_steps, 1)
                print(f"Epoch {epoch + 1}, Step {step + 1}, "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0
                window_steps = 0
        
        # ---------- Compute training epoch metrics ----------
        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_token_acc = (epoch_correct_tokens / epoch_total_tokens) if epoch_total_tokens > 0 else 0.0
        avg_grad_norm = sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1) if epoch_grad_norms else 0.0
        
        # GPU memory usage (measured right after training, before validation)
        # This captures memory during active training, not after cleanup
        gpu_memory_mb = None
        gpu_memory_peak_mb = None
        if torch.cuda.is_available():
            # Allocated memory at this point (after training loop)
            gpu_memory_mb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
            # Peak memory during training (reset was called at start of epoch)
            gpu_memory_peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch time
        epoch_time = time.time() - epoch_start_time

        # ---------- Validation loop ----------
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_correct_tokens = 0
        val_total_tokens = 0
        val_exact_correct = 0
        val_total_sequences = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE, non_blocking=True)
                input_tokens = batch['input_tokens'].to(DEVICE)
                target_tokens = batch['target_tokens'].to(DEVICE)

                logits = model(images, input_tokens)
                # Keep original shape for exact match computation
                batch_size, seq_len = logits.shape[0], logits.shape[1]
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = target_tokens.view(-1)

                loss = criterion(logits_flat, targets_flat)
                val_loss_sum += loss.item()
                val_steps += 1

                # Token accuracy computation using metrics.py
                preds_flat = logits_flat.argmax(dim=-1)
                batch_token_acc = compute_token_accuracy(preds_flat, targets_flat, PAD_ID)
                # Accumulate for validation average
                non_pad_mask = (targets_flat != PAD_ID)
                val_total_tokens += non_pad_mask.sum().item()
                val_correct_tokens += int(batch_token_acc * non_pad_mask.sum().item())
                
                # Compute exact match incrementally using metrics.py (O(1) memory per batch)
                # Reshape predictions back to (batch, seq_len) for sequence-level comparison
                preds_reshaped = preds_flat.view(batch_size, seq_len)
                
                # Use compute_exact_match from metrics.py on this batch only
                batch_exact_match = compute_exact_match(
                    preds_reshaped, target_tokens, PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID
                )
                # Accumulate counts: batch_exact_match is a fraction, multiply by batch_size to get count
                val_exact_correct += int(batch_exact_match * batch_size)
                val_total_sequences += batch_size

        val_avg_loss = val_loss_sum / max(val_steps, 1)
        val_token_acc = (val_correct_tokens / val_total_tokens) if val_total_tokens > 0 else 0.0
        val_exact_match = (val_exact_correct / val_total_sequences) if val_total_sequences > 0 else 0.0

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

        # ---------- Store metrics in dict for future use ----------
        epoch_metrics = {
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'epoch_time_seconds': epoch_time,
            'gradient_norm': avg_grad_norm,
            'gpu_memory_mb': gpu_memory_mb,
            'gpu_memory_peak_mb': gpu_memory_peak_mb,
            'train_loss': epoch_avg_loss,
            'train_token_accuracy': epoch_token_acc,
            'val_loss': val_avg_loss,
            'val_token_accuracy': val_token_acc,
            'val_exact_match': val_exact_match,
        }

        # ---------- Log epoch summary ----------
        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} completed.")
        print(f"  Learning rate: {current_lr:.6f}")
        print(f"  Epoch time: {epoch_time:.2f}s")
        print(f"  Gradient norm: {avg_grad_norm:.4f}")
        if gpu_memory_mb is not None:
            print(f"  GPU memory: {gpu_memory_mb:.1f} MB (peak: {gpu_memory_peak_mb:.1f} MB)")
        print(f"  Train  - loss: {epoch_avg_loss:.4f}, token accuracy: {epoch_token_acc:.4f}")
        print(f"  Val    - loss: {val_avg_loss:.4f}, token accuracy: {val_token_acc:.4f}, exact match: {val_exact_match:.4f}")
        if is_best:
            print(f"  Checkpoint: new best model saved to {best_checkpoint_path}")
        print(f"  Checkpoint: last model saved to {last_checkpoint_path}\n")


if __name__ == "__main__":
    train()

