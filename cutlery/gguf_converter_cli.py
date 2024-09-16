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

def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensor models to GGUF format")
    parser.add_argument("--input_dir", type=str, default="safetensors", help="Directory containing SafeTensor files")
    parser.add_argument("--output_dir", type=str, default="gguf", help="Directory to save GGUF files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all .safetensors files in the input directory
    for safetensor_file in input_dir.glob("*.safetensors"):
        gguf_file = output_dir / (safetensor_file.stem + ".gguf")
        convert_safetensor_to_gguf(str(safetensor_file), str(gguf_file))

if __name__ == "__main__":
    main()