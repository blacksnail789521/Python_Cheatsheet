import os


number_of_processes = len( os.popen("wmic process get description, processid").read().strip().split("\n\n") ) - 1
print(f"number_of_processes: {number_of_processes}")