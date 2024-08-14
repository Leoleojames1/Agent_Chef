from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import pandas as pd
import logging
from Agent_Chef_Class import Agent_Chef_Class
from cutlery.Template_Manager_Class import Template_Manager_Class
from cutlery.Ollama_Interface_Class import Ollama_Interface_Class
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
logging.basicConfig(level=logging.INFO)

# Initialize Agent_Chef, TemplateManager, and OllamaInterface
chef = Agent_Chef_Class()
template_manager = Template_Manager_Class(chef.input_dir)
ollama_interface = Ollama_Interface_Class(None)  # Initialize with no specific model

@app.route('/api/files', methods=['GET'])
def get_files():
    ingredient_files = [f for f in os.listdir(chef.input_dir) if f.endswith(('.json', '.parquet'))]
    dish_files = [f for f in os.listdir(chef.output_dir) if f.endswith('.parquet')]
    construction_files = [f for f in os.listdir(chef.construction_zone_dir) if f.endswith('.txt')]
    latex_files = [f for f in os.listdir(chef.latex_library_dir) if f.endswith('.tex')]
    return jsonify({
        "ingredient_files": ingredient_files,
        "dish_files": dish_files,
        "construction_files": construction_files,
        "latex_files": latex_files
    })

@app.route('/api/file/<type>/<filename>', methods=['GET'])
def get_file_content(type, filename):
    if type == 'ingredient':
        file_path = os.path.join(chef.input_dir, filename)
    elif type == 'dish':
        file_path = os.path.join(chef.output_dir, filename)
    elif type == 'latex':
        file_path = os.path.join(chef.latex_library_dir, filename)
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        if filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            content = df.head(100).to_dict('records')  # Convert first 100 rows to list of dictionaries
            columns = df.columns.tolist()
            return jsonify({"content": content, "columns": columns, "total_rows": len(df)})
        else:
            with open(file_path, 'r') as f:
                content = f.read()
            return jsonify({"content": content})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save', methods=['POST'])
def save_file():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    file_type = data.get('type')
    
    try:
        if not filename or content is None or not file_type:
            raise ValueError("Missing required data")

        if file_type in ['ingredient', 'text']:
            file_path = os.path.join(chef.input_dir, filename)
        elif file_type == 'dish':
            file_path = os.path.join(chef.output_dir, filename)
        else:
            raise ValueError(f"Invalid file type: {file_type}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
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

@app.route('/api/generate_synthetic', methods=['POST'])
def generate_synthetic():
    data = request.json
    seed_parquet = data['seed_parquet']
    num_samples = data['num_samples']
    ollama_model = data['ollama_model']
    system_prompt = data['system_prompt']
    
    try:
        chef.initialize(ollama_model)
        result = chef.run(
            mode='custom',
            seed_parquet=seed_parquet,
            num_samples=num_samples,
            system_prompt=system_prompt
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        else:
            return jsonify({'message': 'Synthetic data generated', 'filename': result['file']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/convert_to_json_parquet', methods=['POST'])
def convert_to_json_parquet():
    data = request.json
    content = data['content']
    template_name = data.get('template')
    filename = data['filename']
    
    try:
        if not template_name:
            return jsonify({'error': 'Template name is required'}), 400

        # Use the Dataset_Manager to parse the text content
        df = chef.dataset_manager.parse_text_to_parquet(content, template_name)
        
        # Generate base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        # Save as JSON
        json_file = os.path.join(chef.input_dir, f"{base_filename}.json")
        df.to_json(json_file, orient='records', indent=2)
        
        # Save as Parquet
        parquet_file = os.path.join(chef.input_dir, f"{base_filename}.parquet")
        df.to_parquet(parquet_file, engine='pyarrow')
        
        return jsonify({
            'json_file': os.path.basename(json_file),
            'parquet_file': os.path.basename(parquet_file)
        })
    except Exception as e:
        logging.exception(f"Error converting to JSON/Parquet: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/run', methods=['POST'])
def run_agent_chef():
    data = request.json
    ollama_model = data.get('ollamaModel')
    
    logging.info(f"Received run request with data: {data}")
    
    try:
        if not ollama_model:
            raise ValueError("Ollama model not specified")
        
        chef.initialize(ollama_model)
        
        seed_parquet = data.get('seedParquet')
        if not seed_parquet:
            raise ValueError("Seed parquet file not specified")
        
        result = chef.run(
            mode=data['mode'],
            seed_parquet=seed_parquet,
            synthetic_technique=data.get('syntheticTechnique'),
            template=data.get('template'),
            system_prompt=data.get('systemPrompt'),
            num_samples=int(data.get('numSamples', 100))
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        else:
            return jsonify({
                'message': result['message'],
                'filename': result['file']
            })
    except Exception as e:
        error_msg = f"Error in run_agent_chef: {str(e)}"
        logging.exception(error_msg)
        return jsonify({"error": error_msg}), 500
    
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
        seed_files = [f for f in os.listdir(chef.input_dir) if f.endswith(('.json', '.parquet'))]
        seed_files.sort(key=lambda x: os.path.getmtime(os.path.join(chef.input_dir, x)), reverse=True)
        
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
    
    if not files:
        return jsonify({'error': 'No files specified for combination'}), 400
    
    try:
        combined_data = []
        for file in files:
            file_path = os.path.join(chef.input_dir, file)
            if file.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                combined_data.append(df)
            elif file.endswith('.json'):
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                df = pd.DataFrame(json_data)
                combined_data.append(df)
            else:  # Assume it's a text file
                with open(file_path, 'r') as f:
                    text_data = f.read()
                df = pd.DataFrame({'content': [text_data]})
                combined_data.append(df)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(chef.input_dir, f'combined_seed_{timestamp}.parquet')
        combined_df.to_parquet(output_file, engine='pyarrow')
        
        return jsonify({
            'message': 'Files combined successfully',
            'parquet_file': os.path.basename(output_file)
        })
    except Exception as e:
        logging.exception(f"Error combining files: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)