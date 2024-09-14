import os
import torch
import subprocess
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_path):
    try:
        state_dict = load_file(model_path)
        print("Model loaded successfully")
        return state_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def merge_adapter(base_model_path, adapter_path, output_path):
    print(f"Loading base model from: {base_model_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading adapter from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return

    print("Merging adapter with base model")
    try:
        merged_model = model.merge_and_unload()
    except Exception as e:
        print(f"Error merging adapter: {e}")
        return

    print(f"Saving merged model to: {output_path}")
    try:
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("Merged model saved successfully")
    except Exception as e:
        print(f"Error saving merged model: {e}")

def convert_to_gguf(input_path, output_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sh_file = os.path.join(script_dir, "safetensors_to_gguf.sh")
    
    if not os.path.exists(sh_file):
        print(f"Error: safetensors_to_gguf.sh not found in {script_dir}")
        return
    
    model_name = os.path.basename(input_path)
    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    
    subprocess.run(["bash", sh_file, gguf_dir, model_name], check=True)
    print(f"Conversion to GGUF completed. Output saved in {gguf_dir}")

if __name__ == "__main__":
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/huggingface_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    adapter_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/checkpoint-900"
    output_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/900"

    print("Select an option:")
    print("1. Merge adapter")
    print("2. Convert safetensor to GGUF")
    print("3. Merge adapter and convert to GGUF")
    
    choice = input("Enter your choice (1/2/3): ")

    if choice in ['1', '3']:
        # First, try to load the base model using safetensors
        model_path = os.path.join(base_model_path, "model.safetensors")
        state_dict = load_model(model_path)

        if state_dict is not None:
            print("Base model loaded successfully using safetensors")
        else:
            print("Failed to load base model using safetensors, proceeding with merge attempt")

        # Attempt to merge the adapter
        merge_adapter(base_model_path, adapter_path, output_path)

    if choice in ['2', '3']:
        input_path = output_path if choice == '3' else base_model_path
        output_dir = os.path.dirname(input_path)
        convert_to_gguf(input_path, output_dir)

    print("Process completed.")