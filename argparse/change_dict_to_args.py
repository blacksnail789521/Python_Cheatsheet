import argparse


def change_dict_to_args(params):
    parser = argparse.ArgumentParser()
    for key in params.keys():
        parser.add_argument("--" + key)

    # create a list of command line args from the dictionary
    args_list = [
        item
        for sublist in [["--" + k, v] for k, v in params.items()]
        for item in sublist
    ]

    # parse the args
    args = parser.parse_args(args_list)
    return args


params = {"arg1": "value1", "arg2": "value2"}
args = change_dict_to_args(params)
print(args)
