import torch
from safetensors.torch import load_file

model_path = "/home/borch/Agent_Chef/agent_chef_data/huggingface_models/Meta-Llama-3.1-8B-bnb-4bit/model.safetensors"
try:
    state_dict = load_file(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")