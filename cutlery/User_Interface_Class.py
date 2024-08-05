from colorama import Fore

class User_Interface_Class:
    def get_user_input(self, prompt):
        return input(Fore.YELLOW + prompt).strip()

    def display_message(self, message, color=Fore.WHITE):
        print(color + message)