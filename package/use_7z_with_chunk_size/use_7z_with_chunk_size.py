import subprocess
import configparser
from os.path import join, basename, splitext
from os import listdir, rename, remove


def run_cmd(cmd):
    
    # Run the cmd.
    proc = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE, \
                                stderr = subprocess.PIPE, universal_newlines = True)
    
    # Print the information.
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line != "":
            print(line, end = "")
        
        line = proc.stderr.readline()
        if line != "":
            print(line, end = "")


def use_7z_with_chunk_size(mode, _7z_path, input_path, output_path = None, chunksize = None):
        
    if mode == "zip":
        # Run the cmd and print the information.
        cmd = [ _7z_path, "a", join(output_path, basename(input_path) + ".zip"), \
                input_path, "-v{}".format(chunksize) ]
        run_cmd(cmd)
        
        # Rename the file_name. (code.zip.001 -> code_001.zip)
        for file_name in listdir(output_path):
            new_file_name, part_number = splitext(file_name)
            new_file_name, ext = splitext(new_file_name)
            new_file_name = new_file_name + "_" + part_number.replace(".", "") + ext
            rename(join(output_path, file_name), \
                      join(output_path, new_file_name))
            
    elif mode == "unzip":
        # Rename the file_name. (code_001.zip -> code.zip.001)
        for file_name in listdir(input_path):
            new_file_name, ext = splitext(file_name)
            new_file_name, part_number = new_file_name.split("_")
            new_file_name = new_file_name + ext + "." + part_number
            rename(join(input_path, file_name), \
                      join(input_path, new_file_name))
        
        # Run the cmd and print the information.
        cmd = [ _7z_path, "x", join(input_path, "*.*"), "-o" + input_path ]
        run_cmd(cmd)
        
        # Remove the zip file.
        for file_name in listdir(input_path):
            if file_name.find("zip") != -1:
                remove( join(input_path, file_name) )


if __name__ == "__main__":
    
    # Set up the parser of ini.
    config = configparser.ConfigParser()
    config.optionxform = str # Avoid changing all config names to lowercase.
    config_file_name = "config.ini"
    config.read(config_file_name)
    
    # Read parameters from ini.
    user = config.get("global", "user")
    mode = config.get("global", "mode")

    _7z_path = config.get(user, "7z_path")
    input_path = config.get(user + "_" + mode, "input_path")
    output_path = config.get(user + "_" + mode, "output_path", fallback = None)
    chunksize = config.get(user + "_" + mode, "chunksize", fallback = None)
    
    # Call use_7z_with_chunk_size.
    use_7z_with_chunk_size(mode, _7z_path, input_path, output_path, chunksize)