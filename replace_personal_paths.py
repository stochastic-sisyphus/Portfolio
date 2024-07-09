import os
import re

# Define the path to your GitHub repository (use the output from the pwd command)
repo_path = '/workspaces/your-repo-name'  # Replace with your actual repo path

# Define the string to replace
replace_string = '/file/path'

# Pattern to match various personal paths starting with /Users or /Users/student
pattern = r'(/Users|/Users/student)[^\'")]*'

# Function to replace strings in a file
def replace_in_file(file_path, pattern, replace_string):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    
    new_contents = re.sub(pattern, replace_string, file_contents)
    
    with open(file_path, 'w') as file:
        file.write(new_contents)

# Walk through the repository
for root, dirs, files in os.walk(repo_path):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        replace_in_file(file_path, pattern, replace_string)

print("Replacement complete.")
