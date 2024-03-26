import json
import yaml

with open("training1.json", "r") as json_file:
    data = json.load(json_file)

# Convert Python dictionary to YAML string
yaml_data = yaml.dump(data, sort_keys=False)

with open("training1.yaml", "w") as yaml_file:
    yaml_file.write(yaml_data)