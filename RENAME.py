#python script to rename large amount of files at once 

import os

# Set the folder path
folder_path = r'C:\Users\nicho\CSE485\BeeCapstone\dataset\images' 

# Loop through all files
for filename in os.listdir(folder_path):
    if filename.startswith('.'):
        continue  # skip hidden files like .DS_Store
    old_file = os.path.join(folder_path, filename)
    
    # Remove the first 9 characters
    new_filename = filename[9:]
    new_file = os.path.join(folder_path, new_filename)
    
    # Rename the file
    os.rename(old_file, new_file)

    #check if its actually doing its job 
    num = 1
    num = num+1
    print(num)

print("Renaming complete!")
