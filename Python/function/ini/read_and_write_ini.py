import configparser


# Set up the parser of ini.
config = configparser.ConfigParser(comment_prefixes = "/", allow_no_value = True) # Keep comments.
config.optionxform = str # Avoid changing all config names to lowercase.
config_file_path = "config.ini"
config.read(config_file_path)

# Read parameters from ini.
some_string = config.get("test", "some_string")
some_int = config.getint("test", "some_int")
some_float = config.getfloat("test", "some_float")
some_bool = config.getboolean("test", "some_bool")
something_not_exist = config.get("test", "something_not_exist", fallback = None)

for element in [some_string, some_int, some_float, some_bool, something_not_exist]:
    print(f"{element}, type = {type(element)}")

# Update ini.
config["test"]["new_stuff"] = "123"
config["test"]["some_int"] = str(some_int + 1)
config.write(open(config_file_path, "w"), space_around_delimiters = True)