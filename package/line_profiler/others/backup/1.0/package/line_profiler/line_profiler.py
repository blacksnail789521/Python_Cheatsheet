import io
import inspect
from os.path import join, abspath, dirname, basename
import sys

package_path = join(dirname(abspath(__file__)))
if package_path not in sys.path: sys.path.insert(0, package_path)

from pprofile import pprofile



def format_output(string, period, default_blacklist):

    def contain_default_blacklist(source_code):
        contain = False
        for element in default_blacklist:
            if source_code.find(element) != -1:
                contain = True
        return contain
    
    """ Phase 1. """
    string_out = io.StringIO()
    skip_this_file = False
    file_time_dict = {}
    total_duration, current_file_path = None, None
    
    for index, line in enumerate(string.split("\n")):
        
        if len(line) == 0:
            # We're at the bottom of the whole file.
            break
        
        if skip_this_file == True and line.startswith("File:") == False:
            continue
        
        
        if line.startswith("Total duration:"):
            # We're at the top of the whole file.
            total_duration = float(line[16:-1])
            print(line, file = string_out)
        
        elif line.startswith("File:"):
            # Skip the file starts with "File: <...>"
            if line.startswith("File: <"):
                skip_this_file = True
                continue
            else:
                skip_this_file = False
            # We're at the top of the single file.
            current_file_path = line[6:]
            file_time_dict[current_file_path] = 0
            print(line, file = string_out)
            
        elif line.startswith("File duration:"):
            print("File duration (Own): {current_file_total} s " + \
                  "({current_file_percentage} %)", \
                  file = string_out)
        
        elif line.startswith("Line #"):
            print("Line #|  Time (s)|  Own|  Own (%)|Source code", \
                  file = string_out)
            
        elif line.startswith("------"):
            print("------+----------+-----+---------+-----------", \
                  file = string_out)
            
        elif line.startswith("(call)") or line.lstrip()[0].isdigit():
            # Get line_number, time and source_code.
            for index, element in enumerate( line.split("|") ):
                if index == 0:
                    # Line #
                    line_number = element
                elif index == 1:
                    # Time (Original: Hits)
                    time = int( element.lstrip() ) * period
                elif index == 5:
                    # Source code
                    source_code = element
            
            # Determine own.
            if line_number == "(call)" and source_code.endswith("runfile"):
                own = "X"
            else:
                if line_number != "(call)" or \
                ( line_number == "(call)" and \
                  contain_default_blacklist(source_code) == True ):
                    own = "V"
                else:
                    own = "X"
                    
            # Initial own_percentage.
            if own == "V":
                own_percentage = "{} %"
            else:
                own_percentage = "X"
            
            # Format "Line #|  Time (s)|  Own|  Own (%)|Source code".
            print("|".join([ line_number, "{:.1f}".format(time).rjust(10), \
                             own.rjust(5), own_percentage.rjust(9), \
                             source_code ]), \
                  file = string_out)
            
            # Add time into corresponing file_time_dict.
            if own == "V":
                file_time_dict[current_file_path] = \
                file_time_dict[current_file_path] + time
    
    string = string_out.getvalue()
    string_out.close()
    
    """ Phase 2. """
    # Update the file duration and line's time percentage.
    string_out = io.StringIO()
    
    for index, line in enumerate(string.split("\n")):
        
        if len(line) == 0:
            # We're at the bottom of the whole file.
            break
        
        if line.startswith("File:"):
            # We're at the top of the single file.
            current_file_path = line[6:]
        
        elif line.startswith("File duration (Own):"):
            # Infuse current file's time.
            current_file_total = file_time_dict[current_file_path]
            current_file_percentage = current_file_total / total_duration * 100
            line = line.replace("{current_file_total}", "{:.1f}" \
                                .format(current_file_total))
            line = line.replace("{current_file_percentage}", "{:.1f}" \
                                .format(current_file_percentage))
        
        elif line.startswith("(call)") or line.lstrip()[0].isdigit():
            # Split line into line_list.
            line_list = line.split("|")
            
            # We only infuse if own == "V".
            own = line_list[2].lstrip()
            if own == "V":
                # Get time.
                time = float( line_list[1].lstrip() )
            
                # Infuse current line's time precentage.
                if file_time_dict[current_file_path] == 0.0:
                    own_percentage = 0.0
                else:
                    own_percentage = time / file_time_dict[current_file_path] * 100
                line_list[3] = line_list[3] \
                               .replace("{}", "{:.2f}".format(own_percentage)) \
                               .lstrip().rjust(9)
            
            line = "|".join(line_list)
    
        print(line, file = string_out)
    
    string = string_out.getvalue()
    string_out.close()
    
    return string
    

def line_profiler(function, file_name = None, mode = "statistic", period = 0.1, \
                  blacklist = [], default_blacklist = ["site-packages", "Anaconda"]):
    
    # Use profiler to get what we want.
    if mode == "statistic":
        profiler = pprofile.StatisticalProfile()
        # By default, we sample every 100ms.
        # It will correctly reflect the results.
        with profiler(period = period):
            everything_you_want_to_see = function()
    elif mode == "deterministic":
        profiler = pprofile.Profile()
        with profiler():
            everything_you_want_to_see = function()
    
    # Set file_path_set.
    # By default, we don't store the result that contain "site-packages" or 
    # "Anaconda" in the path.
    file_path_set = set()
    blacklist.extend(default_blacklist)
    for file_path in profiler.getFilenameSet():
        abs_file_path = abspath(file_path)
        for index, element in enumerate(blacklist):
            if abs_file_path.find(element) != -1:
                break
            if abs_file_path.find(element) == -1 and index == len(blacklist) - 1:
                file_path_set.add(file_path)
    
    # Store the result into string.
    string_out  = io.StringIO()
    profiler.annotate(string_out, filename = file_path_set)
    string = string_out.getvalue()
    string_out.close()
    
    # Format the output.
    string = format_output(string, period, default_blacklist)
    
    # Output the result of profiler.
    caller_path = abspath(inspect.stack()[1].filename)
    caller_dir = dirname(caller_path)
    if file_name is None:
        file_name = basename(caller_path) \
                    .replace(".py", "")
        file_name = "(line_profiler)_" + file_name
    with io.open(join(caller_dir, file_name + ".txt"), "w", \
                 errors = "replace") as out:
            out.write(string)
    
    return everything_you_want_to_see