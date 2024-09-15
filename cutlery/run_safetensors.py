import os
import torch
from safetensors.torch import load_file, safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_safetensors_model(base_model_path):
    # Load the model using safetensors
    safetensor_path = os.path.join(base_model_path, "model.safetensors")
    print(f"Loading safetensors from {safetensor_path}")
    
    tensors = {}
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    return tensors

def setup_model(base_model_path):
    # Load tokenizer and model for LLaMA 3.1
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16,  # Set to float16 for memory optimization
        load_in_8bit=False,
        device_map="auto"
    )
    
    # Optionally apply PEFT (Parameter-Efficient Fine-Tuning)
    model = PeftModel(base_model)
    
    return model, tokenizer

def chat(model, tokenizer):
    print("Chat is ready! Type your input or 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    # Path to the base model and safetensors files
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/900"
    
    # Load safetensors and initialize model
    loaded_tensors = load_safetensors_model(base_model_path)
    
    # Load tokenizer and model
    model, tokenizer = setup_model(base_model_path)
    
    # Chat flow with the model
    chat(model, tokenizer)
