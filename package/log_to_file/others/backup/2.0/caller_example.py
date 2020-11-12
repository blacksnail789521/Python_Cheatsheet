exec("""try:\n    import os\n    with open(os.path.join("log", """ + 
     """"log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")

from sub_program import test_function



@logger.catch
def main():
    
    number = 10
    test_function(number)
    print("Done")
    test_function(number)


if __name__ == "__main__":
    
    log_to_file()
    everything_you_want_to_see = main()