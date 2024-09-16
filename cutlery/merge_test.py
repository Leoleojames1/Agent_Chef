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

def convert_to_gguf(input_path, output_file):
    llama_cpp_dir = os.path.expanduser("~/llama.cpp")
    convert_script = os.path.join(llama_cpp_dir, "convert.py")
    
    if not os.path.exists(convert_script):
        print(f"Error: convert.py not found at {convert_script}")
        return
    
    command = [
        "python",
        convert_script,
        "--outtype", "f16",  # Use f16 to avoid quantization
        "--outfile", output_file,
        input_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion to GGUF completed. Output saved in {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF conversion: {e}")
        print("Full error message:")
        print(e.output)

if __name__ == "__main__":
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/huggingface_models/llama-3-8b-Instruct-bnb-4bit"
    adapter_path = "/home/borch/Agent_Chef/agent_chef_data/oven/adapters/OARC_Commander_v001_LoRA_unsloama3_4bit/checkpoint-900"
    output_path = "/home/borch/Agent_Chef/agent_chef_data/oven/adapters/OARC_Commander_v001_LoRA_unsloama3_4bit/900"
    
    # Create gguf folder
    oven_dir = os.path.dirname(output_path)
    gguf_dir = os.path.join(oven_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

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
        # Convert merged model to GGUF
        merged_gguf = os.path.join(gguf_dir, "merged_model.gguf")
        convert_to_gguf(output_path, merged_gguf)

    print("Process completed.")