import json

# Read the input data
with open('standard_scenario_08.json', 'r') as f:
    data = json.load(f)

# Transform each item
for item in data:
    item['scenario_folder'] = 'adv_init_state'
    item['scenario_id'] = 8
    item['parameters'] = ['genetic_algorithm', True]

# Write the output
with open('adv_scenario_08.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} items successfully!")