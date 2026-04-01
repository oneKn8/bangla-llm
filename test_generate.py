"""Quick test: generate Bengali text from the pre-trained base model."""

import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM

MODEL_DIR = "training/checkpoints/step-4000"
TOKENIZER = "tokenizer/output/bangla_bpe_32k.model"

prompts = [
    "বাংলাদেশের রাজধানী",
    "আজকের আবহাওয়া খুবই",
    "প্রধানমন্ত্রী বলেছেন যে",
    "ক্রিকেট খেলায় বাংলাদেশ",
]

print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER)

print(f"Loading model from {MODEL_DIR}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

for prompt in prompts:
    token_ids = sp.Encode(prompt, out_type=int)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated = output[0].tolist()
    text = sp.Decode(generated)

    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Output: {text}")
    print()
