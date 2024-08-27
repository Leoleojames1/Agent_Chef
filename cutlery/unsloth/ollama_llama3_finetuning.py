import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import apply_chat_template
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

print('=== Loading Base Model ===')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print('=== Loading and Formatting dataset ===')

# Load the parquet file
df = pd.read_parquet("D:/CodingGit_StorageHDD/Ollama_Custom_Mods/Agent_Chef/cutlery/unsloth/OARC_Commander_v001.parquet")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Apply chat template
chat_template = """Below are some tasks!

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
)

print('=== Training the model ===')
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 500,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print('=== Training ===')

trainer_stats = trainer.train()
print(trainer_stats)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print('=== Saving model ===')

# Save the fine-tuned model
model.save_pretrained("borch_OARC_commander_v001_llama-3-8b") # Local saving
tokenizer.save_pretrained("borch_OARC_commander_v001_llama-3-8b_tokenizer") # Local saving

# Save to q4_k_m GGUF
model.save_pretrained_gguf("borch_OARC_commander_v001_llama-3-8b_q4_k_m", tokenizer, quantization_method = "q4_k_m")

# Optional: Push to Hugging Face Hub
# from huggingface_hub import push_to_hub
# push_to_hub(model, "borch/OARC_commander_v001_llama-3-8b", use_temp_dir=True)
# push_to_hub(tokenizer, "borch/OARC_commander_v001_llama-3-8b", use_temp_dir=True)