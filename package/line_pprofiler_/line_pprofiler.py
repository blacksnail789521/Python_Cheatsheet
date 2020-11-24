# https://github.com/vpelletier/pprofile
from pprofile import pprofile

import io
import inspect
import os


def contain_package_list(source_code, package_list):
    contain = False
    for element in package_list:
        if source_code.find(element) != -1:
            contain = True
            break
    return contain


def update_time_and_caller(string_file, time_dict, total_duration):
    
    file_infos = {"file_path": None, \
                  "own":   {"own_overall_time": None, \
                            "own_overall_percentage": None, \
                            "line": []}, \
                  "total": {"total_overall_time": None, \
                            "line": []}}
    
    # Update the file duration and line's time percentage.
    string_file_out = io.StringIO()
    string_file_list = string_file.split("\n")
    for index, line in enumerate(string_file_list):
        
        if len(line) == 0:
            # We're at the bottom of the whole file.
            break
        
        if line.startswith("File:"):
            # We're at the top of the single file.
            current_file_path = line[6:]
            file_infos["file_path"] = current_file_path
        
        elif line.startswith("File duration (Own):"):
            # Infuse current file's time and percentage (own).
            own_overall_time = time_dict["own"][current_file_path]
            own_overall_percentage = time_dict["own"][current_file_path] \
                                     / total_duration * 100
                                     
            line = line.replace("{own_overall_time}", "{:.1f}" \
                                .format(own_overall_time))
            line = line.replace("{own_overall_percentage}", "{:.2f}" \
                                .format(own_overall_percentage))
            
            file_infos["own"]["own_overall_time"] = own_overall_time
            file_infos["own"]["own_overall_percentage"] = \
                own_overall_percentage
            
        elif line.startswith("File duration (Total):"):
            # Infuse current file's time (total).
            total_overall_time = time_dict["total"][current_file_path]
            
            line = line.replace("{total_overall_time}", "{:.1f}" \
                                .format(total_overall_time))
            
            file_infos["total"]["total_overall_time"] = total_overall_time
        
        elif line.lstrip().startswith("(call)") or line.lstrip()[0].isdigit():
            # Split line into line_list.
            line_list = line.split("|")
            
            # Update the "(call)"-like line's line_number and source_code
            # to its caller's. Use a pair of parentheses as well.
            if line_list[0].lstrip() == "(call)":
                find_caller_index = index - 1
                while string_file_list[find_caller_index] \
                      .lstrip().startswith("(call)"):
                    find_caller_index = find_caller_index - 1
                caller_line_list = \
                    string_file_list[find_caller_index].split("|")
                caller_line_number = caller_line_list[0].lstrip()
                line_list[0] = \
                    line_list[0].replace("call", caller_line_number).rjust(8)
                caller_source_code = caller_line_list[4]
                line_list[4] = \
                    line_list[4] + " ( " + caller_source_code.lstrip() + " )"
            
            # Get time.
            time = float( line_list[1].lstrip() )
            
            # We only infuse if own_percentage != "X".
            own_percentage = line_list[2].lstrip()
            if own_percentage != "X":
                # Infuse current line's precentage (own).
                if time_dict["own"][current_file_path] == 0.0:
                    own_percentage = 0.0
                else:
                    own_percentage = \
                        time / time_dict["own"][current_file_path] * 100
                line_list[2] = \
                    line_list[2].replace("{own_percentage}", \
                                         "{:.2f}".format(own_percentage)) \
                                .lstrip().rjust(9)
            
             # We only infuse if total_percentage != "X".
            total_percentage = line_list[3].lstrip()
            if total_percentage != "X":
                # Infuse current line's precentage (total).
                if time_dict["total"][current_file_path] == 0.0:
                    total_percentage = 0.0
                else:
                    total_percentage = \
                        time / time_dict["total"][current_file_path] * 100
                line_list[3] = \
                    line_list[3].replace("{total_percentage}", \
                                         "{:.2f}".format(total_percentage)) \
                                .lstrip().rjust(9)
            
            line = "|".join(line_list)
            
            # Get line_number and source_code.
            line_number = line_list[0].lstrip()
            source_code = line_list[4]
            
            for part in ["own", "total"]:
                file_infos[part]["line"] \
                    .append( {"line_number": line_number, \
                              "time": time, \
                              part + "_percentage": eval(part + \
                                                         "_percentage"), \
                              "source_code": source_code} )

                        
        # It's for every condition.
        print(line, file = string_file_out)
    
    string_file = string_file_out.getvalue()
    string_file_out.close()
    
#    raise Exception(file_infos)
    
    return string_file, file_infos


def update_ranking(ranking, file_infos, max_line_number, max_ranking_number):
    
    for part in ["own", "total"]:
        # We don't need to add the file whose overall_time is zero.
        if file_infos[part][part + "_overall_time"] == 0:
            continue
        
        # Sort the line inside the file_infos 
        # and get the top-"max_line_number".
        file_infos[part]["line"] = \
            [ line for line in file_infos[part]["line"] \
              if line.get(part + "_percentage") != "X" and \
                 line.get(part + "_percentage") != 0 ]
        file_infos[part]["line"] = \
            sorted(file_infos[part]["line"], \
                   key = lambda k: k[part + "_percentage"], \
                   reverse = True)[ : max_line_number]
            
        # Insert the infos into ranking and get the top-"max_ranking_number".
        if part == "own":
            ranking_infos = \
                {"file_path": file_infos["file_path"], \
                "time": file_infos[part][part + "_overall_time"], \
                "percentage": file_infos[part][part + "_overall_percentage"], \
                "line": file_infos[part]["line"]}
        elif part == "total":
            ranking_infos = \
                {"file_path": file_infos["file_path"], \
                "time": file_infos[part][part + "_overall_time"], \
                "line": file_infos[part]["line"]}
        ranking[part].append(ranking_infos)
        ranking[part] = sorted(ranking[part], \
                                key = lambda k: k["time"], \
                                reverse = True)[ : max_ranking_number]
        

def format_output_and_get_ranking(string, period, \
                                  default_blacklist, whitelist, \
                                  max_line_number, max_ranking_number):
    
    string_out = io.StringIO()
    string_out_file = io.StringIO()
    time_dict = {"own": {}, "total": {}}
    ranking = {"own": [], "total": []}
    skip_this_file, total_duration = False, None
    current_file_path, main_appear = None, False
    first_update_time = True
    
    for index, line in enumerate(string.split("\n")):
        
        if len(line) == 0:
            # We're at the bottom of the whole file.
            # Update the own and total time and percentage.
            string_file = string_out_file.getvalue()
            string_out_file.close()
            string_file, file_infos = \
                update_time_and_caller(string_file, time_dict, total_duration)
            update_ranking(ranking, file_infos, \
                           max_line_number, max_ranking_number)
            print(string_file, file = string_out)
            break
        
        if skip_this_file == True and line.startswith("File:") == False:
            continue
        
        # Skip the file starts with "File: <...>"
        if line.startswith("File: <"):
            skip_this_file = True
            continue
        else:
            skip_this_file = False
        
        
        if line.startswith("Total duration:"):
            # We're at the top of the whole file.
            total_duration = float(line[16:-1])
            print(line + "\n", file = string_out)
        
        elif line.startswith("File:"):
            if first_update_time == True:
                first_update_time = False
                string_out_file.close()
            else:
                # Update the own and total time and percentage.
                string_file = string_out_file.getvalue()
                string_out_file.close()
                string_file, file_infos = \
                    update_time_and_caller(string_file, time_dict, \
                                           total_duration)
                update_ranking(ranking, file_infos, \
                               max_line_number, max_ranking_number)
                print(string_file, file = string_out)
                
            # We're at the top of the single file.
            string_out_file = io.StringIO()
            current_file_path = line[6:]
            main_appear = False
            time_dict["own"][current_file_path] = 0
            time_dict["total"][current_file_path] = 0
            print(line, file = string_out_file)
            
        elif line.startswith("File duration:"):
            print("File duration (Own): {own_overall_time} s" + \
                  " ({own_overall_percentage} %)", \
                  file = string_out_file)
            print("File duration (Total): {total_overall_time} s" , \
                  file = string_out_file)
        
        elif line.startswith("Line #"):
            print("  Line #|  Time (s)|  Own (%)|Total (%)|Source code", \
                  file = string_out_file)
            
        elif line.startswith("------"):
            print("--------+----------+---------+---------+-----------", \
                  file = string_out_file)
            
        elif line.lstrip().startswith("(call)") or line.lstrip()[0].isdigit():
            # Get line_number, time and source_code.
            for index, element in enumerate( line.split("|") ):
                if index == 0:
                    # Line #
                    line_number = element.lstrip()
                elif index == 1:
                    # Time (Original: Hits)
                    time = int( element.lstrip() ) * period
                elif index == 5:
                    # Source code
                    source_code = element
                    
                    # Determine main_appear. (It's for "total".)
                    if source_code.find("if __name__") != -1 and \
                       source_code.find("__main__") != -1:
                        main_appear = True
            
            # Determine own and total.
            if line_number == "(call)" and source_code.endswith("runfile"):
                own = "X"
                total = "X"
            else:
                # Determine own.
                if line_number != "(call)" or \
                   ( line_number == "(call)" and \
                     contain_package_list(source_code, \
                                          default_blacklist) and \
                     not contain_package_list(source_code, \
                                              whitelist)\
                     == True ):
                    own = "V"
                else:
                    own = "X"
                
                # Determine total.
                if own == "V":
                    total = "V"
                else:
                    if source_code.find(current_file_path) != -1 or \
                       main_appear == True:
                        total = "X"
                    else:
                        total = "V"
                    
            # Initial own_percentage and total_percentage.
            if own == "V":
                own_percentage = "{own_percentage} %"
            else:
                own_percentage = "X"
            if total == "V":
                total_percentage = "{total_percentage} %"
            else:
                total_percentage = "X"
            
            # Format "  Line #|  Time (s)|  Own (%)|Total (%)|Source code".
            print("|".join([ line_number.rjust(8), \
                             "{:.1f}".format(time).rjust(10), \
                             own_percentage.rjust(9), \
                             total_percentage.rjust(9), \
                             source_code ]), \
                  file = string_out_file)
            
            # Add time into corresponing time_dict.
            if own == "V":
                time_dict["own"][current_file_path] = \
                    time_dict["own"][current_file_path] + time
            if total == "V":
                time_dict["total"][current_file_path] = \
                    time_dict["total"][current_file_path] + time
    
    string = string_out.getvalue()
    string_out.close()
    
    # Remove the last empty line
    string = string[:-1]

    return string, ranking


def output_ranking_to_txt(ranking, caller_dir, file_name):
    
    os.makedirs(os.path.join(caller_dir, "line_pprofiler"), exist_ok = True)
    for part_name, part in ranking.items():
        with io.open( os.path.join(caller_dir, "line_pprofiler", file_name + "_" + part_name + ".log"), \
                      "w", errors = "replace" ) as out:
            for index, file in enumerate(part):
                # Write file path.
                out.write("File: " + file["file_path"] + "\n")
                
                # Write file duration (time + percentage).
                out.write("File duration (" + part_name.capitalize() + \
                          "): " + "{:.1f}".format(file["time"]) + " s ")
                if part_name == "own":
                    out.write("(" + "{:.2f}".format(file["percentage"]) + \
                              " %)" + "\n")
                else:
                    out.write("\n")
                
                # Write line title and demarcation.
                out.write("  Line #|  Time (s)|" + \
                          part_name.capitalize().rjust(5) + " (%)" + \
                          "|Source code" + "\n")
                out.write("--------+----------+---------+-----------" + "\n")
                
                # Write the lines.
                for line in file["line"]:
                    out.write(line["line_number"].rjust(8) + "|" + \
                              "{:.1f}".format(line["time"]).rjust(10) + "|" + \
                              "{:.2f}".format(line[part_name + \
                                   "_percentage"]).rjust(7) + " %" + "|" + \
                              line["source_code"] + "\n")
                
                # We need to change line between different files.
                if index != len(part) - 1:
                    out.write("\n")
    

def line_pprofiler(function, enable = True, file_name = None, \
                   blacklist = ["__package__"], whitelist = [], \
                   default_blacklist = ["site-packages", "Anaconda", "anaconda"], \
                   max_line_number = None, max_ranking_number = None):
    
    # If we don't want to use profiler.
    if enable == False:
        return function()

    # Set pprofile's profiler.
    profiler = pprofile.StatisticalProfile()
    
    # By default, we sample every 100ms.
    period = 0.1
    with profiler(period = period):
        output_of_function = function()

    # Set file_path_set.
        # By default, we don't store the result that contain "site-packages" or 
        # "Anaconda" or "anaconda" in the path.
    file_path_set = set()
    blacklist.extend(default_blacklist)
    for file_path in profiler.getFilenameSet():
        
        abs_file_path = os.path.abspath(file_path)
        
        already_added = False
        
        # Consider whitelist.
        for index, element_for_white in enumerate(whitelist):
            if abs_file_path.find(element_for_white) != -1:
                file_path_set.add(file_path)
                already_added = True
        
        if already_added == True:
            continue
        
        # Consider blacklist.
        for index, element_for_black in enumerate(blacklist):
            if abs_file_path.find(element_for_black) != -1:
                break
            if abs_file_path.find(element_for_black) == -1 and \
               index == len(blacklist) - 1:
                file_path_set.add(file_path)
    
    # Store the result into string.
    string_out  = io.StringIO()
    profiler.annotate(string_out, filename = file_path_set)
    string = string_out.getvalue()
    string_out.close()
    
    # Format the output and get the ranking.
    string, ranking = \
        format_output_and_get_ranking(string, period, \
                                      default_blacklist, whitelist, \
                                      max_line_number, max_ranking_number)
    
    # Output the result of profiler.
    caller_path = os.path.abspath(inspect.stack()[1].filename)
    caller_dir = os.path.dirname(caller_path)
    if file_name is None:
        file_name = os.path.basename(caller_path).replace(".py", "")
        file_name = "(line_pprofiler)_" + file_name
    os.makedirs(os.path.join(caller_dir, "line_pprofiler"), exist_ok = True)
    with io.open( os.path.join(caller_dir, "line_pprofiler", file_name + ".log"), "w", \
                  errors = "replace" ) as out:
            out.write(string)
    
    # Output ranking to txt.
    output_ranking_to_txt(ranking, caller_dir, file_name)
    
    return output_of_function, ranking