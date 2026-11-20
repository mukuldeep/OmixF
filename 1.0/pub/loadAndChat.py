# load model
import torch
import torch.nn as nn
import torch.nn.functional as F
from BigramLanguageModel import BigramLanguageModel
from Omixf1Tokenizer import Omixf1Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "model.pth",
    map_location=device,
    weights_only=True
)

# Restore configuration
config = checkpoint["config"]

vocab_size = config["vocab_size"]
n_embed = config["n_embed"]
block_size = config["block_size"]
n_head = config["n_head"]
n_layer = config["n_layer"]

# Restore tokenizer
# stoi = checkpoint["stoi"]
# itos = checkpoint["itos"]

tokenizer = Omixf1Tokenizer(max_length=7)
vocab_size = tokenizer.vocab_size

# Recreate model
model = BigramLanguageModel().to(device)

# Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully!")

def encode(text):
    return tokenizer.encode(text)

def decode(tokens):
  #print(tokens)
  return ' '.join([tokenizer.decode([i]) for i in tokens])
  #return tokenizer.decode(tokens)


def chat(prompt, max_new_tokens=30):
    #print("prompt:",prompt)

    model.eval()

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_new_tokens
        )

    generated = decode(output[0].tolist())

    return generated

prompt = "G_GGAC"
for i in range(30):
  res = chat(prompt,1)
  print(res)
  prompt = res.replace(' ','')

# print("Model Chat Interface")
# print("Type 'exit' to quit.")

# while True:
#     prompt = input("")
#     if prompt.lower() == "exit":
#         break

#     response = chat(prompt, max_new_tokens=1)
#     print("\nModel:", response)  
