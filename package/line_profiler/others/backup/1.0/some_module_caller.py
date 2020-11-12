exec("""import os\ntry:\n    with open(os.path.join("package", """ + 
     """ "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")
exec("""import os\ntry:\n    with open(os.path.join("package", """ + 
     """ "line_profiler", "import.txt")) as f: exec(f.read())\nexcept:pass""")

from some_module import test_function
import time


def main():
    
    number = 10
    test_function(number)
    print("Done")
    
    return "Done"

if __name__ == "__main__":
    
#    start= time.time()
#    everything_you_want_to_see = main()
#    end = time.time()
#    print("Total time:", "{:.2f}".format(end - start), "s")
    start = time.time()
    everything_you_want_to_see = line_profiler( lambda: log_to_file(lambda : main()), \
                                                blacklist = ["package"] )
    end = time.time()
    del print
    print("{:.2f}".format(end - start))
#    everything_you_want_to_see = line_profiler(lambda : main())