import torch
from torch import nn
from src.models.gpt import GPTModel

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context

    ###Input batch:
 ###tensor([[6109, 3626, 6100,  345],
        ##[6109, 1110, 6622,  257]])
    
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")

# model = GPTModel(GPT_CONFIG_124M)
# model.eval() #A
# #model = GPTModel(GPT_CONFIG_124M)
# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
# print("encoded_tensor.shape:", encoded_tensor.shape)
# out = generate_text_simple(
# model=model,
# idx=encoded_tensor,
# max_new_tokens=6,
# context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output:", out)
# print("Output length:", len(out[0]))

# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
# print("encoded_tensor.shape:", encoded_tensor.shape)

# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)