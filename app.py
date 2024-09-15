from flask import Flask, jsonify, request
from flask_cors import CORS
from colorama import init, Fore, Back, Style
from datetime import datetime
import os
import json
import pandas as pd
import logging
import traceback
import time
from cutlery.DatasetKitchen import DatasetManager, TemplateManager
from cutlery.OllamaInterface import OllamaInterface
from cutlery.FileHandler import FileHandler
from cutlery.UnslothTrainer import UnslothTrainer

init(autoreset=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
logging.basicConfig(level=logging.INFO)

base_dir = os.path.join(os.path.dirname(__file__), 'agent_chef_data')
input_dir = os.path.join(base_dir, "ingredients")
output_dir = os.path.join(base_dir, "dishes")
latex_library_dir = os.path.join(base_dir, "latex_library")
huggingface_dir = os.path.join(base_dir, "huggingface_models")
salad_dir = os.path.join(base_dir, "salad")
oven_dir = os.path.join(base_dir, "oven")
edits_dir = os.path.join(base_dir, "edits")

for dir_path in [huggingface_dir, salad_dir, oven_dir, edits_dir]:
    os.makedirs(dir_path, exist_ok=True)

ollama_interface = OllamaInterface(None)
file_handler = FileHandler(input_dir, output_dir)
template_manager = TemplateManager(input_dir)
dataset_manager = DatasetManager(ollama_interface, template_manager, input_dir, output_dir)

CUSTOM_PROMPTS_DIR = os.path.join(base_dir, 'custom_prompts')
os.makedirs(CUSTOM_PROMPTS_DIR, exist_ok=True)

def initialize(model):
    ollama_interface.set_model(model)

def run(mode, seed_file, **kwargs):
    print(f"{Fore.CYAN}Running with mode: {mode}, seed_file: {seed_file}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Custom prompts: {kwargs.get('custom_prompts', {})}{Style.RESET_ALL}")

    if not seed_file:
        return {'error': "No seed file specified"}

    try:
        if mode == 'custom':
            sample_rate = kwargs.get('sample_rate', 100)
            paraphrases_per_sample = kwargs.get('paraphrases_per_sample', 1)
            column_types = kwargs.get('column_types', {})
            use_all_samples = kwargs.get('use_all_samples', True)
            custom_prompts = kwargs.get('custom_prompts', {})

            seed_file_path = os.path.join(input_dir, seed_file)
            if not os.path.exists(seed_file_path):
                print(f"{Fore.RED}Seed file not found: {seed_file_path}{Style.RESET_ALL}")
                return {'error': f"Seed file not found: {seed_file_path}"}

            print(f"{Fore.GREEN}Generating synthetic data...{Style.RESET_ALL}")
            result_df = dataset_manager.generate_synthetic_data(
                seed_file,
                sample_rate=sample_rate,
                paraphrases_per_sample=paraphrases_per_sample,
                column_types=column_types,
                use_all_samples=use_all_samples,
                custom_prompts=custom_prompts
            )

            if result_df.empty:
                print(f"{Fore.RED}Generated dataset is empty. Check the logs for details.{Style.RESET_ALL}")
                return {'error': "Generated dataset is empty. Check the logs for details."}

            # Generate output filename based on input filename
            input_name = os.path.splitext(seed_file)[0]
            timestamp = int(time.time())
            output_filename = f'{input_name}_synthetic_{timestamp}.parquet'
            output_file = os.path.join(output_dir, output_filename)
            result_df.to_parquet(output_file)

            print(f"{Fore.GREEN}Custom synthetic dataset generated successfully{Style.RESET_ALL}")
            return {
                'message': "Custom synthetic dataset generated successfully",
                'file': output_filename
            }
        else:
            print(f"{Fore.RED}Invalid mode selected{Style.RESET_ALL}")
            return {'error': "Invalid mode selected"}

    except Exception as e:
        error_msg = f"Error in run: {str(e)}\n{traceback.format_exc()}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return {'error': error_msg}

@app.route('/')
def index():
    return "Welcome to AgentChef API"

@app.route('/api/files', methods=['GET'])
def get_files():
    def get_files_from_dir(directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(('.json', '.parquet', '.txt', '.tex'))]

    ingredient_files = get_files_from_dir(input_dir)
    dish_files = get_files_from_dir(output_dir)
    latex_files = [f for f in os.listdir(latex_library_dir) if f.endswith('.tex')]
    salad_files = get_files_from_dir(salad_dir)
    huggingface_folders = [f for f in os.listdir(huggingface_dir) if os.path.isdir(os.path.join(huggingface_dir, f))]
    oven_files = get_files_from_dir(oven_dir)
    edits_files = get_files_from_dir(edits_dir)
    
    return jsonify({
        "ingredient_files": ingredient_files,
        "dish_files": dish_files,
        "latex_files": latex_files,
        "salad_files": salad_files,
        "huggingface_folders": huggingface_folders,
        "oven_files": oven_files,
        "edits_files": edits_files
    })

@app.route('/api/parquet_data', methods=['GET'])
def get_parquet_data():
    filename = request.args.get('filename')
    page = int(request.args.get('page', 0))
    rows_per_page = int(request.args.get('rows_per_page', 10))

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    # Handle 'edits/' prefix
    if filename.startswith('edits/'):
        filename = filename.replace('edits/', '', 1)
        file_path = os.path.join(edits_dir, filename)
    else:
        # Check input_dir, output_dir, salad_dir, and edits_dir for the file
        for dir_path in [input_dir, output_dir, salad_dir, edits_dir]:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                break
        else:
            return jsonify({"error": f"File not found: {filename}. Searched in {input_dir}, {output_dir}, {salad_dir}, {edits_dir}"}), 404

    try:
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        df = pd.read_parquet(file_path)
        columns = df.columns.tolist()
        start = page * rows_per_page
        end = min(start + rows_per_page, len(df))
        page_data = df.iloc[start:end].to_dict('records')
        return jsonify({
            "content": page_data,
            "total_rows": len(df),
            "columns": columns
        })
    except Exception as e:
        return jsonify({"error": f"Error reading parquet file: {str(e)}"}), 500

@app.route('/api/file/<type>/<path:filename>', methods=['GET'])
def get_file_content(type, filename):
    logging.info(f"Received request for file: {filename} of type: {type}")
    if type == 'ingredient':
        file_path = os.path.join(input_dir, filename)
    elif type == 'dish':
        file_path = os.path.join(output_dir, filename)
    elif type == 'latex':
        file_path = os.path.join(latex_library_dir, filename)
    elif type == 'salad':
        file_path = os.path.join(salad_dir, filename)
    elif type == 'edit':
        # Handle files in the 'edits' folder
        file_path = os.path.join(input_dir, 'edits', filename)
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        if filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            content = df.head(100).to_dict('records')  # Convert first 100 rows to list of dictionaries
            columns = df.columns.tolist()
            return jsonify({"content": content, "columns": columns, "total_rows": len(df)})
        elif filename.endswith(('.txt', '.json', '.tex')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({"content": content})
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 500

@app.route('/api/file/edit/<path:filename>', methods=['GET'])
def get_edit_file_content(filename):
    file_path = os.path.join(input_dir, 'edits', filename)
    return _read_file_content(file_path, filename)

def _read_file_content(file_path, filename):
    try:
        if filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            content = df.head(100).to_dict('records')  # Convert first 100 rows to list of dictionaries
            columns = df.columns.tolist()
            return jsonify({"content": content, "columns": columns, "total_rows": len(df)})
        elif filename.endswith(('.txt', '.json', '.tex')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({"content": content})
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 500

@app.route('/api/save', methods=['POST'])
def save_file():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    
    try:
        if not filename or content is None:
            raise ValueError("Missing required data")

        # Always save as txt file
        base_filename = os.path.splitext(filename)[0]
        txt_filename = f"{base_filename}.txt"
        file_path = os.path.join(input_dir, txt_filename)
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        
        return jsonify({"success": True, "message": "File saved successfully", "file_path": file_path})
    except Exception as e:
        logging.exception(f"Error saving file: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    templates = template_manager.get_templates()
    print("Sending templates:", templates)  # Add this line for debugging
    return jsonify(templates)

@app.route('/api/templates', methods=['POST'])
def add_template():
    data = request.json
    template_name = data.get('name')
    template_fields = data.get('fields')
    if not template_name or not template_fields:
        return jsonify({'success': False, 'message': 'Template name and fields are required'}), 400
    try:
        template_manager.add_template(template_name, template_fields)
        return jsonify({'success': True, 'message': 'Template added successfully'})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/template/<template_name>', methods=['GET'])
def load_template(template_name):
    template = template_manager.load_template(f"{template_name}.json")
    if template is None:
        return jsonify({'error': 'Template not found'}), 404
    return jsonify(template)

@app.route('/api/template/create', methods=['POST'])
def create_template():
    data = request.json
    template_data = data.get('template_data')
    filename = data.get('filename')
    if not template_data or not filename:
        return jsonify({'error': 'Template data and filename are required'}), 400
    template = template_manager.create_template(template_data, filename)
    return jsonify(template)

@app.route('/api/save_template', methods=['POST'])
def save_template():
    data = request.json
    name = data.get('name')
    columns = data.get('columns')
    
    if not name or not columns:
        return jsonify({'error': 'Template name and columns are required'}), 400
    try:
        template_data = {name: columns}
        template = template_manager.create_template(template_data, f"{name}.json")
        return jsonify({'success': True, 'message': 'Template saved successfully', 'template': template})
    except Exception as e:
        return jsonify({'error': f'Failed to save template: {str(e)}'}), 500
    
@app.route('/api/run', methods=['POST'])
def run_agent_chef():
    data = request.json
    print(f"{Fore.CYAN}Received data: {json.dumps(data, indent=2)}{Style.RESET_ALL}")
    ollama_model = data.get('ollamaModel')
    custom_chat_template = data.get('customChatTemplate')

    try:
        if not ollama_model:
            print(f"{Fore.RED}Ollama model not specified{Style.RESET_ALL}")
            raise ValueError("Ollama model not specified")

        print(f"{Fore.GREEN}Initializing Ollama model: {ollama_model}{Style.RESET_ALL}")
        initialize(ollama_model)

        seed_file = data.get('seedFile')
        if not seed_file:
            print(f"{Fore.RED}Seed file not specified{Style.RESET_ALL}")
            raise ValueError("Seed file not specified")

        # Ensure the seed file exists
        seed_file_path = os.path.join(input_dir, seed_file)
        if not os.path.exists(seed_file_path):
            print(f"{Fore.RED}Seed file not found: {seed_file_path}{Style.RESET_ALL}")
            raise ValueError(f"Seed file not found: {seed_file_path}")

        print(f"{Fore.GREEN}Running with seed file: {seed_file}{Style.RESET_ALL}")
        
        # If a custom chat template is provided, use it
        if custom_chat_template:
            print(f"{Fore.GREEN}Using custom chat template{Style.RESET_ALL}")
            # You'll need to implement a function to apply the custom template
            # This could be part of your UnslothTrainer or a separate utility
            apply_custom_chat_template(custom_chat_template)

        result = run(
            mode='custom',
            seed_file=seed_file,
            sample_rate=data.get('sampleRate', 100),
            paraphrases_per_sample=data.get('paraphrasesPerSample', 1),
            column_types=data.get('columnTypes', {}),
            use_all_samples=data.get('useAllSamples', True),
            custom_prompts=data.get('customPrompts', {})
        )

        if 'error' in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return jsonify({'error': result['error']}), 400
        else:
            print(f"{Fore.GREEN}Success: {result['message']}{Style.RESET_ALL}")
            return jsonify({
                'message': result['message'],
                'filename': result['file']
            })
    except Exception as e:
        error_msg = f"Error in run_agent_chef: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
@app.route('/api/generate_synthetic', methods=['POST'])
def generate_synthetic():
    data = request.json
    seed_parquet = data['seed_parquet']
    num_samples = data['num_samples']
    ollama_model = data['ollama_model']
    system_prompt = data['system_prompt']
    
    try:
        initialize(ollama_model)
        result = run(
            mode='custom',
            seed_file=seed_parquet,
            sample_rate=100,
            paraphrases_per_sample=num_samples,
            custom_prompts={'system': system_prompt}
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        else:
            return jsonify({'message': 'Synthetic data generated', 'filename': result['file']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/convert_to_json', methods=['POST'])
def convert_to_json():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    template_name = data.get('template')
    file_type = data.get('fileType')
    
    try:
        if not filename or not template_name:
            return jsonify({'error': 'Filename and template name are required'}), 400

        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        # Use the content from the request if available, otherwise read from file
        if not content:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        # Use the Dataset_Manager to parse the content to JSON
        df, json_file, parquet_file = dataset_manager.parse_text_to_parquet(content, template_name, os.path.splitext(filename)[0])
        
        return jsonify({
            'message': 'JSON and Parquet files created successfully',
            'json_file': os.path.basename(json_file),
            'parquet_file': os.path.basename(parquet_file)
        })
    except Exception as e:
        logging.exception(f"Error converting to JSON: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/parse_dataset', methods=['POST'])
def parse_dataset():
    data = request.json
    content = data.get('content')
    template_name = data.get('template')
    mode = data.get('mode', 'manual')
    
    try:
        if not content or not template_name:
            return jsonify({'error': 'Content and template name are required'}), 400

        df = dataset_manager.parse_dataset(content, template_name, mode)
        
        # Save as JSON
        json_filename = f"parsed_dataset_{int(time.time())}.json"
        json_file = os.path.join(input_dir, json_filename)
        df.to_json(json_file, orient='records', indent=2)

        # Save as Parquet
        parquet_filename = f"parsed_dataset_{int(time.time())}.parquet"
        parquet_file = os.path.join(input_dir, parquet_filename)
        df.to_parquet(parquet_file, engine='pyarrow')

        return jsonify({
            'success': True,
            'message': 'Dataset parsed successfully',
            'json_file': json_filename,
            'parquet_file': parquet_filename,
            'result': df.to_dict(orient='records')
        })
    except Exception as e:
        logging.exception(f"Error parsing dataset: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
    
@app.route('/api/convert_to_json_parquet', methods=['POST'])
def convert_to_json_parquet():
    data = request.json
    filename = data['filename']
    
    try:
        json_file_path = os.path.join(input_dir, filename)
        if not os.path.exists(json_file_path):
            return jsonify({'error': 'JSON file not found'}), 404

        # Read JSON file
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data)

        # Generate base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        # Save as Parquet
        parquet_file = os.path.join(input_dir, f"{base_filename}.parquet")
        df.to_parquet(parquet_file, engine='pyarrow')
        
        return jsonify({
            'message': 'Parquet seed created successfully',
            'parquet_file': os.path.basename(parquet_file)
        })
    except Exception as e:
        logging.exception(f"Error converting to Parquet: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
@app.route('/api/slice_parquet', methods=['POST'])
def slice_parquet():
    data = request.json
    filename = data.get('filename')
    columns_to_remove = data.get('columns_to_remove', [])

    if not filename or not columns_to_remove:
        return jsonify({"error": "Filename and columns to remove are required"}), 400

    try:
        # Find the file in input_dir, output_dir, or salad_dir
        for dir_path in [input_dir, output_dir, salad_dir]:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                break
        else:
            return jsonify({"error": f"File not found: {filename}"}), 404

        df = pd.read_parquet(file_path)
        
        # Ensure columns_to_remove only contains columns that exist in the DataFrame
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        # Remove the selected columns
        df = df.drop(columns=columns_to_remove)

        new_filename = f"{os.path.splitext(filename)[0]}_sliced.parquet"
        new_file_path = os.path.join(edits_dir, new_filename)
        df.to_parquet(new_file_path, engine='pyarrow')

        return jsonify({
            "message": f"Sliced parquet saved as {new_filename} in 'edits' directory",
            "removed_columns": columns_to_remove
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/save_parquet_edits', methods=['POST'])
def save_parquet_edits():
    data = request.json
    filename = data.get('filename')
    edits = data.get('edits', {})

    if not filename or not edits:
        return jsonify({"error": "Filename and edits are required"}), 400

    try:
        # Find the file in input_dir, output_dir, or salad_dir
        for dir_path in [input_dir, output_dir, salad_dir]:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                break
        else:
            return jsonify({"error": f"File not found: {filename}"}), 404

        df = pd.read_parquet(file_path)

        for row_index, row_edits in edits.items():
            for column, value in row_edits.items():
                df.at[int(row_index), column] = value

        df.to_parquet(file_path, engine='pyarrow')

        return jsonify({"message": "Edits saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/save_parquet_as_new', methods=['POST'])
def save_parquet_as_new():
    data = request.json
    filename = data.get('filename')
    edits = data.get('edits', {})

    if not filename or not edits:
        return jsonify({"error": "Filename and edits are required"}), 400

    try:
        # Find the file in input_dir, output_dir, or salad_dir
        for dir_path in [input_dir, output_dir, salad_dir]:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                break
        else:
            return jsonify({"error": f"File not found: {filename}"}), 404

        df = pd.read_parquet(file_path)

        for row_index, row_edits in edits.items():
            for column, value in row_edits.items():
                df.at[int(row_index), column] = value

        new_filename = f"{os.path.splitext(filename)[0]}_edited.parquet"
        new_file_path = os.path.join(edits_dir, new_filename)
        df.to_parquet(new_file_path, engine='pyarrow')

        return jsonify({"message": f"Edits saved as new file: {new_filename} in 'edits' directory"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/generate_paraphrases', methods=['POST'])
def generate_paraphrases():
    data = request.json
    sample = data['sample']
    num_paraphrases = data['num_paraphrases']
    ollama_model = data['ollama_model']
    
    try:
        initialize(ollama_model)
        paraphrases = dataset_manager.enhanced_generator.generate_paraphrases(sample, num_paraphrases)
        return jsonify({'paraphrases': paraphrases})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/api/ollama-models', methods=['GET'])
def get_ollama_models():
    try:
        models = ollama_interface.list_models()
        return jsonify({"models": models})
    except Exception as e:
        logging.exception("Error fetching Ollama models")
        return jsonify({"error": str(e), "message": "Error fetching Ollama models"}), 500
    
@app.route('/api/seeds', methods=['GET'])
def get_seeds():
    try:
        seed_files = [f for f in os.listdir(input_dir) if f.endswith(('.json', '.parquet'))]
        seed_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
        
        if not seed_files:
            return jsonify({"seeds": [], "message": "No seed files found"}), 200
        
        most_recent_seed = seed_files[0] if seed_files else None
        
        return jsonify({
            "seeds": seed_files,
            "most_recent_seed": most_recent_seed,
            "message": "Seeds fetched successfully"
        }), 200
    except Exception as e:
        logging.exception("Error fetching seeds")
        return jsonify({"error": str(e), "message": "Error fetching seeds"}), 500
    
@app.route('/api/combine_files', methods=['POST'])
def combine_files():
    data = request.json
    files = data.get('files', [])
    
    if len(files) < 2:
        return jsonify({'error': 'At least two files must be selected for combination'}), 400
    
    # Check if all files are of the same type
    file_types = set(os.path.splitext(file['name'])[1] for file in files)
    if len(file_types) > 1:
        return jsonify({'error': 'All selected files must be of the same type'}), 400

    file_extension = file_types.pop()  # Get the file extension (including the dot)
    
    try:
        # Create a new filename based on input file names
        base_names = [os.path.splitext(file['name'])[0] for file in files]
        new_filename = '_'.join(base_names)
        
        if file_extension == '.parquet':
            # Special handling for parquet files
            dfs = []
            for file in files:
                file_name = file['name']
                file_type = file['type']
                
                if file_type == 'ingredient':
                    file_path = os.path.join(input_dir, file_name)
                elif file_type == 'dish':
                    file_path = os.path.join(output_dir, file_name)
                else:
                    return jsonify({'error': f'Invalid file type: {file_type}'}), 400
                
                df = pd.read_parquet(file_path)
                print(f"{Fore.CYAN}Read file: {file_name}, Shape: {df.shape}{Style.RESET_ALL}")
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"{Fore.GREEN}Combined DataFrame Shape: {combined_df.shape}{Style.RESET_ALL}")
            
            # Save the combined data
            output_filename = f'{new_filename}.parquet'
            output_file = os.path.join(salad_dir, output_filename)
            combined_df.to_parquet(output_file, engine='pyarrow')
            print(f"{Fore.GREEN}Saved combined file: {output_file}{Style.RESET_ALL}")
        else:
            # Handling for text-based files (txt, json, tex)
            combined_data = []
            for file in files:
                file_name = file['name']
                file_type = file['type']
                
                if file_type == 'ingredient':
                    file_path = os.path.join(input_dir, file_name)
                elif file_type == 'dish':
                    file_path = os.path.join(output_dir, file_name)
                else:
                    return jsonify({'error': f'Invalid file type: {file_type}'}), 400
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    combined_data.append(content)
                print(f"{Fore.CYAN}Read file: {file_name}, Length: {len(content)}{Style.RESET_ALL}")
            
            # Combine the data
            combined_content = '\n\n'.join(combined_data)
            
            # Save the combined data
            output_filename = f'{new_filename}{file_extension}'
            output_file = os.path.join(salad_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            print(f"{Fore.GREEN}Saved combined file: {output_file}{Style.RESET_ALL}")
        
        return jsonify({
            'message': 'Files combined successfully',
            'combined_file': output_filename
        })
    except Exception as e:
        error_msg = f"Error combining files: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({'error': error_msg}), 500
    
@app.route('/api/save_prompt_set', methods=['POST'])
def save_prompt_set():
    data = request.json
    name = data.get('name')
    prompts = data.get('prompts')
    
    if not name or not prompts:
        return jsonify({'error': 'Name and prompts are required'}), 400
    
    filename = os.path.join(CUSTOM_PROMPTS_DIR, f"{name}.json")
    with open(filename, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    return jsonify({'message': f'Prompt set "{name}" saved successfully'})

@app.route('/api/load_prompt_set/<name>', methods=['GET'])
def load_prompt_set(name):
    filename = os.path.join(CUSTOM_PROMPTS_DIR, f"{name}.json")
    try:
        with open(filename, 'r') as f:
            prompts = json.load(f)
        return jsonify(prompts)
    except FileNotFoundError:
        return jsonify({'error': 'Prompt set not found'}), 404

@app.route('/api/list_prompt_sets', methods=['GET'])
def list_prompt_sets():
    files = [f[:-5] for f in os.listdir(CUSTOM_PROMPTS_DIR) if f.endswith('.json')]
    return jsonify(files)

@app.route('/api/unsloth_train', methods=['POST'])
def unsloth_train():
    data = request.json
    print(f"{Fore.CYAN}Received Unsloth training data: {json.dumps(data, indent=2)}{Style.RESET_ALL}")
    
    training_file = data.get('trainingFile')
    validation_file = data.get('validationFile')
    test_file = data.get('testFile')
    huggingface_model = data.get('huggingfaceModel')
    new_model_name = data.get('newModelName')
    num_train_epochs = data.get('numTrainEpochs', 1)
    per_device_train_batch_size = data.get('perDeviceTrainBatchSize', 2)
    gradient_accumulation_steps = data.get('gradientAccumulationSteps', 4)
    precision = data.get('precision', '4bit')

    try:
        if not training_file or not huggingface_model:
            raise ValueError("Training file and Hugging Face model must be specified")

        unsloth_trainer = UnslothTrainer(base_dir, input_dir, oven_dir)
        
        result = unsloth_trainer.train(
            model_name=os.path.join(huggingface_dir, huggingface_model),
            train_dataset=os.path.join(input_dir, training_file),
            validation_dataset=os.path.join(input_dir, validation_file) if validation_file else None,
            test_dataset=os.path.join(input_dir, test_file) if test_file else None,
            output_name=new_model_name,
            max_steps=num_train_epochs * 100,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            load_in_4bit=(precision == '4bit'),
            load_in_16bit=(precision == '16bit')
        )

        unsloth_trainer.cleanup()

        return jsonify({
            'message': 'Unsloth training completed successfully',
            'training_result': result
        })

    except Exception as e:
        error_msg = f"Error in Unsloth training: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
@app.route('/api/unsloth_generate', methods=['POST'])
def unsloth_generate():
    data = request.json
    model_dir = data.get('model_dir')
    prompt = data.get('prompt')
    max_new_tokens = data.get('max_new_tokens', 128)

    try:
        unsloth_trainer = UnslothTrainer(input_dir, output_dir)
        unsloth_trainer.load_model(model_dir)
        generated_text = unsloth_trainer.generate_text(prompt, max_new_tokens)

        return jsonify({
            'message': 'Text generated successfully',
            'generated_text': generated_text
        })

    except Exception as e:
        error_msg = f"Error in unsloth_generate: {str(e)}"
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
@app.route('/api/adapter-files', methods=['GET'])
def get_adapter_files():
    try:
        adapter_files = []
        oven_models = []
        for root, dirs, files in os.walk(oven_dir):
            for dir in dirs:
                if dir.startswith('checkpoint-'):
                    adapter_files.append(os.path.join(os.path.relpath(root, oven_dir), dir))
                elif os.path.isfile(os.path.join(root, dir, 'config.json')):  # Check if it's a model directory
                    oven_models.append(os.path.join(os.path.relpath(root, oven_dir), dir))
        return jsonify({"adapter_files": adapter_files, "oven_models": oven_models})
    except Exception as e:
        logging.exception("Error fetching adapter and oven model files")
        return jsonify({"error": str(e), "message": "Error fetching adapter and oven model files"}), 500
    
@app.route('/api/convert_to_gguf', methods=['POST'])
def convert_to_gguf():
    data = request.json
    print(f"{Fore.CYAN}Received GGUF conversion data: {json.dumps(data, indent=2)}{Style.RESET_ALL}")
    
    input_path = data.get('inputPath')
    output_name = data.get('outputName')
    outtype = data.get('outtype', 'f16')

    try:
        if not input_path:
            raise ValueError("Input model path must be specified")

        # Construct the full path to the input model's "900" directory
        full_input_path = os.path.join(oven_dir, input_path, "900")
        if not os.path.exists(full_input_path):
            raise FileNotFoundError(f"Input model '900' directory not found: {full_input_path}")

        # Construct the path to the GGUF output directory
        gguf_output_dir = os.path.join(oven_dir, "gguf_models")
        os.makedirs(gguf_output_dir, exist_ok=True)

        # Path to the safetensors_to_GGUF.sh script
        script_path = os.path.join(base_dir, "safetensors_to_GGUF.sh")

        # Construct the command
        command = [
            "bash", script_path,
            gguf_output_dir,
            output_name,
            full_input_path
        ]

        print(f"{Fore.GREEN}Running command: {' '.join(command)}{Style.RESET_ALL}")
        
        # Run the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print(f"{Fore.GREEN}GGUF conversion completed successfully{Style.RESET_ALL}")
        return jsonify({
            'message': 'GGUF conversion completed successfully',
            'output_file': os.path.join(gguf_output_dir, f"{output_name}-q8_0.gguf")
        })

    except subprocess.CalledProcessError as e:
        error_msg = f"Error in GGUF conversion: {e.stderr}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"Error in GGUF conversion: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
@app.route('/api/merge_adapter', methods=['POST'])
def merge_adapter():
    data = request.json
    print(f"{Fore.CYAN}Received Unsloth merge data: {json.dumps(data, indent=2)}{Style.RESET_ALL}")
    
    base_model_path = data.get('baseModelPath')
    adapter_path = data.get('adapterPath')
    output_name = data.get('outputName')
    dequantize = data.get('dequantize', 'no')  # 'no', 'f16', or 'f32'

    try:
        if not base_model_path or not adapter_path or not output_name:
            raise ValueError("Base model, adapter model, and output name must be specified")

        unsloth_trainer = UnslothTrainer(base_dir, input_dir, oven_dir)

        print(f"{Fore.GREEN}Starting Unsloth merge{Style.RESET_ALL}")
        result = unsloth_trainer.merge_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_name=output_name,
            convert_to_gguf=True,
            dequantize=dequantize
        )
        
        return jsonify({
            'message': 'Unsloth merge completed successfully',
            'merge_result': result
        })

    except Exception as e:
        error_msg = f"Error in Unsloth merge: {str(e)}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)