from unsloth import FastLanguageModel
from unsloth import to_sharegpt, standardize_sharegpt
from unsloth import apply_chat_template
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train and use Unsloth models")
    
    parser.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length")
    parser.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization")
    parser.add_argument('--dataset', type=str, help="Dataset to use for training")
    parser.add_argument('--validation_dataset', type=str, help="Validation dataset")
    parser.add_argument('--test_dataset', type=str, help="Test dataset")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training")
    parser.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    parser.add_argument('--quantization', type=str, default="q8_0", help="Quantization method for saving the model")
    
    args = parser.parse_args()

    print('=== Loading Model ===')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print('=== Loading and Formatting Datasets ===')
    train_dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    if args.validation_dataset:
        val_dataset = load_dataset("parquet", data_files=args.validation_dataset, split="train")
    else:
        val_dataset = None

    # Apply chat template if needed
    # train_dataset = apply_chat_template(train_dataset, tokenizer, chat_template)
    # if val_dataset:
    #     val_dataset = apply_chat_template(val_dataset, tokenizer, chat_template)

    print('=== Training the model ===')
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    print('=== Training ===')
    trainer_stats = trainer.train()
    print(trainer_stats)

    print('=== Saving model ===')
    model.save_pretrained(os.path.join(args.output_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    if args.save_gguf:
        print(f'=== Saving GGUF model with {args.quantization} quantization ===')
        gguf_dir = os.path.join(args.output_dir, f"gguf_{args.quantization}")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.quantization)

if __name__ == "__main__":
    main()