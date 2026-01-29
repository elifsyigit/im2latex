import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import DEVICE, MAX_LATEX_LENGTH, SOS_ID, EOS_ID
from data.dataset import Im2LatexDataset
from models.transformer import Im2LatexModel


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    return checkpoint['model_state_dict'], checkpoint['vocab_size'], checkpoint['tokenizer']


def greedy_decode(model, encoder_outputs, max_length=MAX_LATEX_LENGTH):
    batch_size = encoder_outputs.size(0)
    device = encoder_outputs.device
    
    decoded_sequences = []
    
    for b in range(batch_size):
        current_tokens = [SOS_ID]
        encoder_out = encoder_outputs[b:b+1]
        
        for step in range(max_length):
            input_tokens = torch.tensor([current_tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model.decoder(input_tokens, encoder_out)
            
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            
            current_tokens.append(next_token_id)
            
            if next_token_id == EOS_ID:
                break
        
        decoded_sequences.append(current_tokens)
    
    return decoded_sequences


def evaluate():
    checkpoint_path = Path("checkpoints/im2latex_model.pt")
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    print("Loading checkpoint...")
    model_state_dict, vocab_size, tokenizer = load_checkpoint(checkpoint_path)
    print(f"Vocab size: {vocab_size}")
    
    print("Initializing model...")
    model = Im2LatexModel(vocab_size=vocab_size)
    model.load_state_dict(model_state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print("Loading validation dataset...")
    val_dataset = Im2LatexDataset(tokenizer=tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    num_samples = min(10, len(val_dataset))
    print(f"\nEvaluating on {num_samples} samples...\n")
    print("=" * 80)
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= num_samples:
                break
            
            images = batch['image'].to(DEVICE)
            target_tokens = batch['target_tokens'].squeeze(0).cpu().tolist()
            
            encoder_outputs = model.encoder(images)
            
            predicted_token_ids = greedy_decode(model, encoder_outputs, max_length=MAX_LATEX_LENGTH)[0]
            
            ground_truth_latex = tokenizer.decode(target_tokens)
            predicted_latex = tokenizer.decode(predicted_token_ids)
            
            print(f"\nSample {idx + 1}:")
            print(f"Ground Truth: {ground_truth_latex}")
            print(f"Predicted:    {predicted_latex}")
            print("-" * 80)


if __name__ == "__main__":
    evaluate()

