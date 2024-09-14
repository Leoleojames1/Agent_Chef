#!/bin/bash
set -e

gguf_dir="$1"
model_name="$2"
input_dir=$(dirname "$gguf_dir")

cd "$input_dir"
python llama.cpp/convert-hf-to-gguf.py --outtype q8_0 --model-name "$model_name-q8_0" --outfile "$gguf_dir/$model_name-q8_0.gguf" "$input_dir/$model_name"
# Uncomment the following lines if you want to generate f16 and f32 versions as well
# python llama.cpp/convert-hf-to-gguf.py --outtype f16 --model-name "$model_name-f16" --outfile "$gguf_dir/$model_name-f16.gguf" "$input_dir/$model_name"
# python llama.cpp/convert-hf-to-gguf.py --outtype f32 --model-name "$model_name-f32" --outfile "$gguf_dir/$model_name-f32.gguf" "$input_dir/$model_name"