import argparse


def change_dict_to_args(configs: dict) -> argparse.Namespace:
    args = argparse.Namespace()
    for key, value in configs.items():
        setattr(args, key, value)
    return args


configs = {"arg1": "value1", "arg2": "value2", "params": {"A": 1, "B": "2"}}
print(configs)
configs = change_dict_to_args(configs)
print(configs)
