import os
import json

class Template_Manager_Class:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def load_template(self, template_file):
        template_path = os.path.join(self.input_dir, template_file)
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return json.load(f)
        else:
            return None

    def create_template(self, template_data, filename):
        if not filename.endswith('.json'):
            filename += '.json'
        template_path = os.path.join(self.input_dir, filename)
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        return template_data

    def get_templates(self):
        templates = {
            "formula": {
                "formula": "The mathematical formula",
                "explanation": "Explanation of the formula",
                "python_code": "Python code to implement the formula",
                "example": "An example of using the formula"
            },
            "ai_concept": {
                "concept": "Name of the AI concept",
                "definition": "Definition of the concept",
                "use_case": "A common use case for this concept",
                "example": "An example of the concept in action"
            },
            "data_structure": {
                "name": "Name of the data structure",
                "description": "Description of the data structure",
                "time_complexity": "Time complexity for common operations",
                "python_implementation": "Basic Python implementation"
            }
        }
        return templates