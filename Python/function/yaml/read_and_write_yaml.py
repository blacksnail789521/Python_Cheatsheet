import yaml


# Read yaml
with open('config.yaml', 'r') as yaml_file:
    config_dict = yaml.load(yaml_file, Loader = yaml.FullLoader)

# Update yaml
config_dict['new_stuff'] = 'XD'

# Save yaml
with open('new_config.yaml', 'w') as yaml_file:
    yaml.dump(config_dict, yaml_file)