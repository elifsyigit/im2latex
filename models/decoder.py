import torch
import torch.nn as nn
import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_SIZE, NUM_DECODER_LAYERS, NUM_ATTENTION_HEADS, MAX_LATEX_LENGTH, PAD_ID


class LatexDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBEDDING_SIZE, num_layers=NUM_DECODER_LAYERS,
                 num_heads=NUM_ATTENTION_HEADS, max_len=MAX_LATEX_LENGTH):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _create_padding_mask(self, input_tokens):
        return (input_tokens == PAD_ID)
    
    def forward(self, input_tokens, encoder_outputs):
        batch_size, seq_len = input_tokens.size()
        
        padding_mask = self._create_padding_mask(input_tokens)
        
        token_embeds = self.token_embedding(input_tokens)
        
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        
        decoder_input = token_embeds + pos_embeds
        
        causal_mask = self._generate_causal_mask(seq_len, input_tokens.device)
        
        decoder_input = decoder_input.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_outputs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )
        
        decoder_output = decoder_output.transpose(0, 1)
        
        logits = self.output_projection(decoder_output)
        
        return logits


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 2
    seq_len = 50
    num_visual_tokens = 16
    
    decoder = LatexDecoder(vocab_size=vocab_size)
    
    dummy_input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_encoder_outputs = torch.randn(batch_size, num_visual_tokens, EMBEDDING_SIZE)
    
    output_logits = decoder(dummy_input_tokens, dummy_encoder_outputs)
    
    print(f"Input tokens shape: {dummy_input_tokens.shape}")
    print(f"Encoder outputs shape: {dummy_encoder_outputs.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Expected: (batch, seq_len, vocab_size) = ({batch_size}, {seq_len}, {vocab_size})")

