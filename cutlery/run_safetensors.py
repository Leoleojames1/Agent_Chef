import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model

def setup_model(base_model_path, lora_adapter_path):
    # Load tokenizer for LLaMA 3.1
    print(f"Loading base model from {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load the base model with safetensors (automatically handles multi-part safetensors files)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Optimize memory usage
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter configuration and apply it
    print(f"Loading LoRA adapter from {lora_adapter_path}")
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    
    # Apply the LoRA adapter to the base model
    model = PeftModel(base_model, peft_config)

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
    # Path to the base model directory containing safetensors and tokenizer files
    # base_model_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/900"
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/huggingface_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    
    # Path to the LoRA adapter directory
    lora_adapter_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/checkpoint-900"

    # Load the tokenizer and model with the LoRA adapter
    model, tokenizer = setup_model(base_model_path, lora_adapter_path)
    
    # Start chat flow
    chat(model, tokenizer)
