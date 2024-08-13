import os
import json

class Template_Manager_Class:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.templates_file = os.path.join(input_dir, 'templates.json')
        print(f"Templates file path: {self.templates_file}")  # Add this line for debugging
        self.templates = self.load_templates()

    def load_templates(self):
        if os.path.exists(self.templates_file):
            with open(self.templates_file, 'r') as f:
                templates = json.load(f)
        else:
            templates = {
                "instruct": ["user", "request", "assistant", "response"],
                "mathFusion": ["formula", "solution", "python", "example"],
                "latexSeries": ["formula", "solution"],
                "latexTheory": ["theory", "explanation"],
                "aiConcept": ["concept", "definition", "useCase", "example"],
                "dataStructure": ["name", "description", "timeComplexity", "pythonImplementation"],
                "functionCall": ["command", "description", "args", "actions"],
                "pythonBase": ["code", "description", "args", "returns"],
                "pythonOllama": ["code", "description", "args", "actions", "chainOfThought","prompts"],
            }
            self.save_templates(templates)
        
        # Ensure all template values are lists
        for key, value in templates.items():
            if not isinstance(value, list):
                templates[key] = list(value) if isinstance(value, (tuple, set)) else [str(value)]
        
        return templates

    def get_templates(self):
        return self.templates

    def save_templates(self, templates):
        with open(self.templates_file, 'w') as f:
            json.dump(templates, f, indent=2)

    def create_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]

    def get_template(self, template_name):
        return self.templates.get(template_name)

    def add_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]