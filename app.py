from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import pandas as pd
import logging
from Agent_Chef_Class import Agent_Chef_Class
from cutlery.Template_Manager_Class import Template_Manager_Class

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Initialize Agent_Chef and TemplateManager
chef = Agent_Chef_Class()
template_manager = Template_Manager_Class(input_dir='templates')

@app.route('/api/files', methods=['GET'])
def get_files():
    ingredient_files = [f for f in os.listdir(chef.input_dir) if f.endswith('.json')]
    dish_files = [f for f in os.listdir(chef.output_dir) if f.endswith('.parquet')]
    latex_files = [f for f in os.listdir(chef.latex_library_dir) if f.endswith('.tex')]
    return jsonify({
        "ingredient_files": ingredient_files,
        "dish_files": dish_files,
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
    filename = data['filename']
    content = data['content']
    file_type = data['type']
    
    if file_type == 'ingredient':
        file_path = os.path.join(chef.input_dir, filename)
    elif file_type == 'dish':
        file_path = os.path.join(chef.output_dir, filename)
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Convert JSON to Parquet if it's an ingredient file
        if file_type == 'ingredient':
            parquet_path = file_path.replace('.json', '.parquet')
            chef.file_handler.save_json_to_parquet(filename)
        
        logging.info(f"File saved successfully: {file_path}")
        return jsonify({"message": "File saved successfully"})
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/run', methods=['POST'])
def run_agent_chef():
    data = request.json
    chef.mode = data['mode']
    chef.model = data['ollamaModel']
    chef.template = data['template']
    chef.system_prompt = data['systemPrompt']
    chef.user_json = data.get('selectedFile')
    chef.dataset_name = data.get('datasetName')

    try:
        chef.main()
        logging.info("Agent Chef ran successfully")
        return jsonify({"output": "Agent Chef ran successfully"})
    except Exception as e:
        logging.error(f"Error running Agent Chef: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    templates = template_manager.get_templates()
    return jsonify(templates)

@app.route('/api/template/create', methods=['POST'])
def create_template():
    data = request.json
    template_data = data.get('template_data')
    filename = data.get('filename')
    if not template_data or not filename:
        return jsonify({'error': 'Template data and filename are required'}), 400
    template = template_manager.create_template(template_data, filename)
    return jsonify(template)

@app.route('/api/template/<template_file>', methods=['GET'])
def load_template(template_file):
    template = template_manager.load_template(template_file)
    if template is None:
        return jsonify({'error': 'Template not found'}), 404
    return jsonify(template)

if __name__ == '__main__':
    app.run(debug=True)
