#!/bin/bash
set -e

gguf_dir="$1"
model_name="$2"
input_dir="$3"

cd "$input_dir"
python $HOME/llama.cpp/convert_hf_to_gguf.py --outtype q8_0 --model-name "$model_name-q8_0" --outfile "$gguf_dir/$model_name-q8_0.gguf" .
# Uncomment the following lines if you want to generate f16 and f32 versions as well
# python $HOME/llama.cpp/convert_hf_to_gguf.py --outtype f16 --model-name "$model_name-f16" --outfile "$gguf_dir/$model_name-f16.gguf" .
# python $HOME/llama.cpp/convert_hf_to_gguf.py --outtype f32 --model-name "$model_name-f32" --outfile "$gguf_dir/$model_name-f32.gguf" .