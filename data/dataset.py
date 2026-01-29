import kagglehub
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_LATEX_LENGTH, PAD_ID
from data.tokenizer import LaTeXTokenizer
from data.transforms import ResizePad



class Im2LatexDataset(Dataset):
    def __init__(self, dataset_path=None, tokenizer=None, split='train'):
        if dataset_path is None:
            path = kagglehub.dataset_download("shahrukhkhan/im2latex100k")
            self.dataset_path = Path(path)
        else:
            self.dataset_path = Path(dataset_path)
        
        self.split = split
        
        split_map = {
            'train': 'im2latex_train.csv',
            'test': 'im2latex_test.csv',
            'validate': 'im2latex_validate.csv',
            'val': 'im2latex_validate.csv'
        }
        
        csv_file = self.dataset_path / split_map.get(split, f'im2latex_{split}.csv')
        self.annotations = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.annotations.append({
                    'latex': row['formula'],
                    'image': row['image']
                })
        
        if tokenizer is None:
            all_latex = [item['latex'] for item in self.annotations]
            self.tokenizer = LaTeXTokenizer(all_latex)
        else:
            self.tokenizer = tokenizer
        
        self.image_dir = self.dataset_path / 'formula_images_processed'/ 'formula_images_processed'
        
        self.transform = transforms.Compose([
            ResizePad(IMAGE_HEIGHT, IMAGE_WIDTH),
            transforms.ToTensor(),   # gives (1, H, W)
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = self.image_dir / item['image']
        latex_str = item['latex']
        
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image)
        
        token_ids = self.tokenizer.encode(latex_str)
        
        input_tokens = token_ids[:-1]
        target_tokens = token_ids[1:]
        
        input_tokens = self._pad_sequence(input_tokens, MAX_LATEX_LENGTH)
        target_tokens = self._pad_sequence(target_tokens, MAX_LATEX_LENGTH)
        
        return {
            'image': image_tensor,
            'input_tokens': torch.tensor(input_tokens, dtype=torch.long),
            'target_tokens': torch.tensor(target_tokens, dtype=torch.long)
        }
    
    def _pad_sequence(self, sequence, max_length):
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence + [PAD_ID] * (max_length - len(sequence))
        return sequence


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = Im2LatexDataset(split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.tokenizer.vocab_size}")
    
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"Image: {sample['image'].shape}")
    print(f"Input tokens: {sample['input_tokens'].shape}")
    print(f"Target tokens: {sample['target_tokens'].shape}")
    
    print(f"\nInput tokens (first 20): {sample['input_tokens'][:20].tolist()}")
    print(f"Target tokens (first 20): {sample['target_tokens'][:20].tolist()}")

