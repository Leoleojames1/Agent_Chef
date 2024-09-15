import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def setup_model(base_model_path):
    # Load tokenizer and model for LLaMA 3.1
    print(f"Loading model from {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load the model with safetensors
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16,  # Optimize memory
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Optionally apply PEFT (if your model uses PEFT fine-tuning)
    model = PeftModel(model)

    return model, tokenizer

def chat(model, tokenizer):
    print("Chat is ready! Type your input or 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=150)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    # Path to the model directory containing safetensors and tokenizer files
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/900"
    
    # Load the tokenizer and model
    model, tokenizer = setup_model(base_model_path)
    
    # Start chat flow
    chat(model, tokenizer)
