import glob
import os
import re

for f in glob.glob(r'C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\notebooks\*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace relative import path
    content = content.replace("sys.path.append(os.path.abspath(os.path.join('..', 'src')))", "sys.path.append(os.path.abspath('src'))")
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
        
print("Paths fixed in notebooks.")
