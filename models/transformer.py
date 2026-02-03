import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_HEIGHT, IMAGE_WIDTH
from models.encoder import ImageEncoder
from models.decoder import LatexDecoder


class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = LatexDecoder(vocab_size=vocab_size)
    
    def forward(self, images, input_tokens):
        encoder_outputs = self.encoder(images)
        logits = self.decoder(input_tokens, encoder_outputs)
        return logits


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 2
    seq_len = 50
    
    model = Im2LatexModel(vocab_size=vocab_size)
    
    dummy_images = torch.randn(batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    dummy_input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output_logits = model(dummy_images, dummy_input_tokens)
    


