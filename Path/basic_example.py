from pathlib import Path

# Create a Path object for the current directory
current_dir = Path('.')

# List all files and directories in the current directory
print("Contents of the current directory:")
for item in current_dir.iterdir():
    print(item)

# Create a new directory
new_dir = current_dir / 'new_directory'
new_dir.mkdir(exist_ok=True)
print(f"\nCreated new directory: {new_dir}")

# Create a new file in the new directory
new_file = new_dir / 'example.txt'
new_file.write_text("This is an example file created using pathlib.")
print(f"\nCreated new file: {new_file}")

# Read the content of the new file
file_content = new_file.read_text()
print(f"\nContent of {new_file}:")
print(file_content)
