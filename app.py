import io
import torch
from fastapi import FastAPI, File, UploadFile
from models.transformer import Im2LatexModel            
from config import MAX_LATEX_LENGTH
from infer import preprocess_image, greedy_decode

CHECKPOINT_PATH = "checkpoints/im2latex_model.pt"

def load_checkpoint_cpu(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint['model_state_dict'], checkpoint['vocab_size'], checkpoint['tokenizer']

model_state, vocab_size, tokenizer = load_checkpoint_cpu(CHECKPOINT_PATH)

model = Im2LatexModel(vocab_size=vocab_size)
model.load_state_dict(model_state)
model.to(torch.device('cpu'))
model.eval()

app = FastAPI(title="im2latex inference API")

@app.post("/predict")
@torch.no_grad()
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    image_stream = io.BytesIO(image_bytes)
    image_tensor = preprocess_image(image_stream).to(torch.device('cpu'))

    encoder_output = model.encoder(image_tensor)
    predicted_token_ids = greedy_decode(model, encoder_output, max_length=MAX_LATEX_LENGTH)
    
    latex = tokenizer.decode(predicted_token_ids)

    return {
        "latex": latex
    }