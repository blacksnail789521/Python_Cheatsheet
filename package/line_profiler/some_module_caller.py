exec("""from os.path import join, abspath, dirname\ntry:\n    """ +
     """with open(join(abspath(dirname(__file__)), "package", """ + 
     """ "log_to_file", "import.txt")) as f: exec(f.read())\nexcept:pass""")
exec("""from os.path import join, abspath, dirname\ntry:\n    """ +
     """with open(join(abspath(dirname(__file__)), "package", """ + 
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
    everything_you_want_to_see = \
        line_profiler( lambda: log_to_file(lambda : main()), \
                       blacklist = ["package"], output_ranking = True )
    end = time.time()
    del print
    print("{:.5f}".format(end - start))
#    everything_you_want_to_see = line_profiler(lambda : main())