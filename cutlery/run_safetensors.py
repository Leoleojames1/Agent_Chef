import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import gradio as gr

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

def generate_response(user_input, model, tokenizer):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def launch_gradio(model, tokenizer):
    # Chat history to store the conversation
    chat_history = []

    def chat_interface(user_input):
        # Get model response
        response = generate_response(user_input, model, tokenizer)
        
        # Add user input and response to the chat history
        chat_history.append(("User", user_input))
        chat_history.append(("Assistant", response))
        
        # Format the chat history for display
        formatted_history = ""
        for speaker, message in chat_history:
            formatted_history += f"{speaker}: {message}\n\n"
        
        return formatted_history, ""

    def clear_history():
        chat_history.clear()
        return "", ""

    # Create Gradio blocks
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with LLaMA 3.1 Model with LoRA Adapter")
        
        with gr.Column():
            # Text area for the conversation history (read-only)
            chat_output = gr.Textbox(label="Chat History", interactive=False, lines=20, placeholder="Chat will appear here.")
            
            # Input area for user message
            user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1)
            
            # Submit button and Clear button
            with gr.Row():
                submit_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear Chat")
        
        # Bind the submit button and clear button to their respective functions
        submit_btn.click(chat_interface, inputs=user_input, outputs=[chat_output, user_input])
        clear_btn.click(clear_history, outputs=[chat_output, user_input])

    # Launch the Gradio app
    demo.launch()

if __name__ == "__main__":
    # Path to the base model directory containing safetensors and tokenizer files
    base_model_path = "/home/borch/Agent_Chef/agent_chef_data/huggingface_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    
    # Path to the LoRA adapter directory
    lora_adapter_path = "/home/borch/Agent_Chef/agent_chef_data/oven/Meta-Llama-3.1-8B-Instruct-bnb-4bit_OARC_Commander_v001/checkpoint-900"

    # Load the tokenizer and model with the LoRA adapter
    model, tokenizer = setup_model(base_model_path, lora_adapter_path)
    
    # Launch the Gradio interface
    launch_gradio(model, tokenizer)
