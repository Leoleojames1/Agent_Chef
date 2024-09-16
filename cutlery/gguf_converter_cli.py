import argparse
import os
import subprocess
from pathlib import Path

def convert_safetensor_to_gguf(input_file, output_file):
    """
    Convert a SafeTensor file to GGUF format using llama.cpp's convert_to_gguf.py script.
    """
    convert_script = "./convert_to_gguf.py"  # Assuming convert_to_gguf.py is in the same directory
    command = [
        "python", convert_script,
        "--input", input_file,
        "--output", output_file,
        "--outtype", "f16"
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed for {input_file}: {e}")

def process_directory(input_dir, output_dir):
    """
    Process all .safetensors files in the input directory and convert them to GGUF format.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all .safetensors files in the input directory
    for safetensor_file in input_dir.glob("*.safetensors"):
        gguf_file = output_dir / (safetensor_file.stem + ".gguf")
        convert_safetensor_to_gguf(str(safetensor_file), str(gguf_file))

def get_user_input():
    """
    Get input and output directories from user input.
    """
    input_dir = input("Enter the directory containing SafeTensor files (default: safetensors): ").strip() or "safetensors"
    output_dir = input("Enter the directory to save GGUF files (default: gguf): ").strip() or "gguf"
    return input_dir, output_dir

def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensor models to GGUF format")
    parser.add_argument("--input_dir", type=str, help="Directory containing SafeTensor files")
    parser.add_argument("--output_dir", type=str, help="Directory to save GGUF files")
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        # Use command-line arguments if provided
        input_dir = args.input_dir
        output_dir = args.output_dir
    else:
        # If no arguments are provided, use interactive mode
        input_dir, output_dir = get_user_input()

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()