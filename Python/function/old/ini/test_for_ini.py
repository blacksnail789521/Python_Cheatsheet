import configparser



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
    print(user, mode, _7z_path, input_path, output_path, chunksize, sep = "\n")