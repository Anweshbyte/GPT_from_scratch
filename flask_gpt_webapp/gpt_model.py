import torch
import tiktoken
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src.models.gpt import GPTModel  # Now Python can find src

# Load model configuration (same as training)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("/Users/arpitasen/Desktop/GPT/gpt_model.pth", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def generate_text(prompt, max_new_tokens=50):
    """Generate response from the trained GPT model."""
    model.eval()
    
    input_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)
        
        if next_token.item() == 50256:  # Stop on end-of-text token
            break
    
    return tokenizer.decode(input_tensor.squeeze().tolist())