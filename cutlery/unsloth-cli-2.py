#!/usr/bin/env python3

"""
🦥 Enhanced Script for Fine-Tuning FastLanguageModel with Unsloth

This script extends the original unsloth-cli.py with additional features:
- Validation split option for better performance monitoring
- [You can add more features here as needed]

Usage example:
    python unsloth-cli-2.py --model_name "unsloth/llama-3-8b" --dataset "your_dataset" \
    --validation_split 0.1 --max_seq_length 2048 --load_in_4bit \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --max_steps 1000 --learning_rate 2e-5 --output_dir "outputs" \
    --save_model --save_path "model" --quantization "q4_k_m"

To see a full list of configurable options, use:
    python unsloth-cli-2.py --help

Happy fine-tuning!
"""

import argparse
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import logging

logging.getLogger('hf-to-gguf').setLevel(logging.WARNING)

def run(args):
    print('=== Loading Model ===')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    print('=== Loading and Formatting Dataset ===')
    dataset = load_dataset(args.dataset)
    
    if args.validation_split > 0:
        train_test_split = dataset['train'].train_test_split(test_size=args.validation_split)
        dataset = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })
    else:
        dataset = DatasetDict({
            'train': dataset['train']
        })

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            texts.append(text + tokenizer.eos_token)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("Data is formatted and ready!")

    print('=== Configuring Training Arguments ===')
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
    )

    print('=== Initializing Trainer ===')
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print('=== Starting Training ===')
    trainer_stats = trainer.train()
    print(trainer_stats)

    if args.save_model:
        print('=== Saving Model ===')
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=args.quantization,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🦥 Enhanced fine-tuning script using Unsloth")
    
    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization")
    model_group.add_argument('--dataset', type=str, required=True, help="Hugging Face dataset to use for training")
    model_group.add_argument('--validation_split', type=float, default=0.0, help="Percentage of training data to use for validation (0.0 to 0.2)")

    lora_group = parser.add_argument_group("🧠 LoRA Options")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for LoRA model")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout rate")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")

    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps")
    training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility")

    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard", choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"], help="The list of integrations to report the results and logs to")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps")

    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+", help="Quantization method for saving the model")

    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    args = parser.parse_args()
    run(args)