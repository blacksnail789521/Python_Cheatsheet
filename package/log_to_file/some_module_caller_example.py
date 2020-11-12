exec("""import os\ntry:\n    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "package", "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")

from some_module import test_function



def main():
    
    number = 10
    test_function(number)
    print("Done")
    
    return "Done"


if __name__ == "__main__":
    
    everything_you_want_to_see = \
        log_to_file(lambda : main(), DB_OPTION = True, \
                    db_config = {"level": "ERROR", "fab": "F12", "db": "MES", \
                                 "table_name": "MFGDEV3.AUTO_MINE_PIPELINE_LOG_BTH"})