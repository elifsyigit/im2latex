import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import csv


sys.path.append(str(Path(__file__).parent))
from config import BATCH_SIZE, LEARNING_RATE, DEVICE, PAD_ID, SOS_ID, EOS_ID
from data.dataset import Im2LatexDataset
from models.transformer import Im2LatexModel
from evaluation.metrics import compute_token_accuracy, compute_exact_match, compute_edit_distance


def train():
    print("Loading dataset...")
    train_dataset = Im2LatexDataset(split='train')
    val_dataset = Im2LatexDataset(split='val', tokenizer=train_dataset.tokenizer)
    
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
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    metrics_path = log_dir / "metrics.csv"

    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0


    
    print("Initializing model...")
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "im2latex.pt"
    best_checkpoint_path = checkpoint_dir / "im2latex_best.pt"

    model = Im2LatexModel(vocab_size=vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    start_epoch = 0
    
    print("Starting training from scratch...")
    
    model.train()
    num_epochs = 20
    print_interval = 100
    clip_value = 1.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {DEVICE}")
    print(f"Print loss every {print_interval} steps\n")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        
        running_loss = 0.0
        window_steps = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_correct_tokens = 0
        epoch_total_tokens = 0
        epoch_grad_norms_sum = 0.0
        clipping_trigger_count = 0
        
        for step, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE, non_blocking=True)
            input_tokens = batch['input_tokens'].to(DEVICE, non_blocking=True)
            target_tokens = batch['target_tokens'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(images, input_tokens)
            
            logits_flat = logits.view(-1, vocab_size)
            targets = target_tokens.view(-1)

            loss = criterion(logits_flat, targets)
            
            loss.backward()
            
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            epoch_grad_norms_sum += raw_norm.item()
            
            if raw_norm > clip_value:
                clipping_trigger_count += 1
            
            optimizer.step()
            
            batch_loss = loss.item()
            running_loss += batch_loss
            window_steps += 1
            epoch_loss_sum += batch_loss
            epoch_steps += 1

            with torch.no_grad():
                preds_flat = logits_flat.argmax(dim=-1)
                batch_token_acc = compute_token_accuracy(preds_flat, targets, PAD_ID)
                non_pad_mask = (targets != PAD_ID)
                epoch_total_tokens += non_pad_mask.sum().item()
                epoch_correct_tokens += int(batch_token_acc * non_pad_mask.sum().item())
            
            if (step + 1) % print_interval == 0:
                avg_loss = running_loss / max(window_steps, 1)
                print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
                window_steps = 0
        
        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_token_acc = (epoch_correct_tokens / epoch_total_tokens) if epoch_total_tokens > 0 else 0.0
        avg_grad_norm = epoch_grad_norms_sum / max(epoch_steps, 1)
        
        gpu_memory_mb = None
        gpu_memory_peak_mb = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
            gpu_memory_peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_correct_tokens = 0
        val_total_tokens = 0
        val_exact_correct = 0
        val_total_sequences = 0
        val_total_edit_distance = 0.0
        val_total_edit_distance_normalized = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE, non_blocking=True)
                input_tokens = batch['input_tokens'].to(DEVICE)
                target_tokens = batch['target_tokens'].to(DEVICE)

                logits = model(images, input_tokens)
                batch_size, seq_len = logits.shape[0], logits.shape[1]
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = target_tokens.view(-1)

                loss = criterion(logits_flat, targets_flat)
                val_loss_sum += loss.item()
                val_steps += 1

                preds_flat = logits_flat.argmax(dim=-1)
                batch_token_acc = compute_token_accuracy(preds_flat, targets_flat, PAD_ID)
                non_pad_mask = (targets_flat != PAD_ID)
                val_total_tokens += non_pad_mask.sum().item()
                val_correct_tokens += int(batch_token_acc * non_pad_mask.sum().item())
                
                preds_reshaped = preds_flat.view(batch_size, seq_len)
                batch_exact_match = compute_exact_match(
                    preds_reshaped, target_tokens, PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID
                )
                val_exact_correct += int(batch_exact_match * batch_size)
                val_total_sequences += batch_size

                avg_edit_dist, avg_norm_edit_dist = compute_edit_distance(
                    preds_reshaped, target_tokens, train_dataset.tokenizer, 
                    PAD_ID, SOS_ID, EOS_ID
                )
                val_total_edit_distance += avg_edit_dist * batch_size
                val_total_edit_distance_normalized += avg_norm_edit_dist * batch_size

        val_avg_loss = val_loss_sum / max(val_steps, 1)
        val_token_acc = (val_correct_tokens / val_total_tokens) if val_total_tokens > 0 else 0.0
        val_exact_match = (val_exact_correct / val_total_sequences) if val_total_sequences > 0 else 0.0
        val_avg_edit_distance = val_total_edit_distance / val_total_sequences if val_total_sequences > 0 else 0.0
        val_avg_norm_edit_distance = val_total_edit_distance_normalized / val_total_sequences if val_total_sequences > 0 else 0.0

        is_best = val_avg_loss < best_val_loss
        if is_best:
            best_val_loss = val_avg_loss

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_token_acc",
                    "val_loss",
                    "val_token_acc",
                    "val_exact_match",
                    "val_edit_distance",
                    "val_norm_edit_distance",
                    "learning_rate",
                    "grad_norm",
                    "epoch_time_sec",
                    "gpu_memory_mb",
                    "gpu_memory_peak_mb"
                ])
                write_header = False

            writer.writerow([
                epoch + 1,
                epoch_avg_loss,
                epoch_token_acc,
                val_avg_loss,
                val_token_acc,
                val_exact_match,
                val_avg_edit_distance,
                val_avg_norm_edit_distance,
                current_lr,
                avg_grad_norm,
                epoch_time,
                gpu_memory_mb,
                gpu_memory_peak_mb
            ])
            f.flush()



        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "vocab_size": vocab_size,
            "tokenizer": train_dataset.tokenizer,
        }

        torch.save(checkpoint_state, last_checkpoint_path)

        if is_best:
            torch.save(checkpoint_state, best_checkpoint_path)

        model.train()

        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} completed.")
        print(f"  Learning rate: {current_lr:.6f}")
        print(f"  Epoch time: {epoch_time:.2f}s")
        print(f"  Avg Gradient Norm: {avg_grad_norm:.4f} (Clipped {clipping_trigger_count} times)")
        if gpu_memory_mb is not None:
            print(f"  GPU memory: {gpu_memory_mb:.1f} MB (peak: {gpu_memory_peak_mb:.1f} MB)")
        print(f"  Train  - loss: {epoch_avg_loss:.4f}, token accuracy: {epoch_token_acc:.4f}")
        print(f"  Val    - loss: {val_avg_loss:.4f}, token accuracy: {val_token_acc:.4f}, exact match: {val_exact_match:.4f}")
        print(f"  Val    - edit distance: {val_avg_edit_distance:.4f}, normalized: {val_avg_norm_edit_distance:.4f}")
        if is_best:
            print(f"  Checkpoint: new best model saved to {best_checkpoint_path}")
        print(f"  Checkpoint: last model saved to {last_checkpoint_path}\n")


if __name__ == "__main__":
    train()