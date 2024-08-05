import os
from colorama import Fore

class Prompt_Manager_Class:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def load_system_prompt(self):
        prompt_file = input(Fore.YELLOW + "Enter the name of the system prompt file (or press Enter to create a new one): ").strip()
        if prompt_file:
            with open(os.path.join(self.input_dir, prompt_file), 'r') as f:
                return f.read().strip()
        else:
            return self.create_system_prompt()

    def create_system_prompt(self):
        print(Fore.CYAN + "Creating a new system prompt.")
        prompt = input(Fore.YELLOW + "Enter the system prompt: ").strip()
        filename = input(Fore.YELLOW + "Enter a name to save this system prompt: ").strip()
        if not filename.endswith('.txt'):
            filename += '.txt'
        with open(os.path.join(self.input_dir, filename), 'w') as f:
            f.write(prompt)
        print(Fore.GREEN + f"System prompt saved as {filename}")
        return prompt