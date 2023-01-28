import argparse

# Get args
parser = argparse.ArgumentParser(
    description="TimePCL",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default value
)
parser.add_argument("--seed", type=int, help="the random seed", default=42)
parser.add_argument("--batch_size", type=int, help="the batch size", default=128)
args = parser.parse_known_args(args=[])[0]  # Allow unrecognized arguments
print(args)  # Namespace(batch_size=128, seed=42)

# Update args with a dictionary
new_hp_dict = {"seed": 43, "batch_size": 32, "not_exist": "Adam"}
for key, value in new_hp_dict.items():
    setattr(args, key, value)
print(args)  # Namespace(batch_size=32, not_exist='Adam', seed=43)
