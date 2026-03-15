import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import sys
import argparse
import random

sys.path.append(str(Path(__file__).parent))
from config import DEVICE, MAX_LATEX_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, SOS_ID, EOS_ID
from models.transformer import Im2LatexModel
from data.dataset import Im2LatexDataset
from data.transforms import ResizePad


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    return checkpoint['model_state_dict'], checkpoint['vocab_size'], checkpoint['tokenizer']


def preprocess_image(image_path):
    transform = transforms.Compose([
        ResizePad(IMAGE_HEIGHT, IMAGE_WIDTH),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def greedy_decode(model, encoder_output, max_length=MAX_LATEX_LENGTH):
    device = encoder_output.device
    current_tokens = [SOS_ID]
    
    for step in range(max_length):
        input_tokens = torch.tensor([current_tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model.decoder(input_tokens, encoder_output)
        
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
        
        current_tokens.append(next_token_id)
        
        if next_token_id == EOS_ID:
            break
    
    return current_tokens


def infer(image_path, checkpoint_path="checkpoints/im2latex_best.pt", ground_truth_latex=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found at {image_path}")
        return
    
    print("Loading checkpoint...")
    model_state_dict, vocab_size, tokenizer = load_checkpoint(checkpoint_path)
    
    print("Initializing model...")
    model = Im2LatexModel(vocab_size=vocab_size)
    model.load_state_dict(model_state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loading image: {image_path}")
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(DEVICE)
    
    print("Running inference...")
    with torch.no_grad():
        encoder_output = model.encoder(image_tensor)
        predicted_token_ids = greedy_decode(model, encoder_output, max_length=MAX_LATEX_LENGTH)
        predicted_latex = tokenizer.decode(predicted_token_ids)
    
    print(f"\nImage: {image_path}")
    print(f"Predicted LaTeX: {predicted_latex}")
    
    if ground_truth_latex:
        print(f"Ground truth LaTeX: {ground_truth_latex}")
        if predicted_latex == ground_truth_latex:
            print("✓ Prediction matches ground truth!")
        else:
            print("✗ Prediction differs from ground truth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer LaTeX from a random image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/im2latex_best.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validate"],
                        help="Dataset split to pick random image from")
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = Im2LatexDataset(split=args.split)
    
    random_idx = random.randint(0, len(dataset) - 1)
    random_image_name = dataset.annotations[random_idx]['image']
    ground_truth_latex = dataset.annotations[random_idx]['latex']
    image_path = dataset.image_dir / random_image_name
    
    print(f"Selected random image #{random_idx}: {random_image_name}")
    print(f"Ground truth LaTeX: {ground_truth_latex}\n")
    
    infer(image_path, args.checkpoint, ground_truth_latex)

