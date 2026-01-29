import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_SIZE):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.projection = nn.Linear(512, embed_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        features = self.cnn(x)
        
        batch, channels, height, width = features.size()
        num_tokens = height * width
        
        features = features.view(batch, channels, num_tokens)
        features = features.permute(0, 2, 1)
        
        output = self.projection(features)
        
        return output


if __name__ == "__main__":
    encoder = ImageEncoder()
    
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    output = encoder(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch, num_tokens, embed_dim) = ({batch_size}, num_tokens, {EMBEDDING_SIZE})")
    print(f"Actual: {output.shape}")

