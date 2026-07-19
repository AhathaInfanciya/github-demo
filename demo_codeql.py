# demo_codeql.py

import subprocess

user_input = input("Enter command: ")

subprocess.run(user_input, shell=True)