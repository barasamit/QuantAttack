import json


# Open the text file and read its contents
with open("/dt/shabtaia/dt-fujitsu/datasets/others/imagnet/LOC_synset_mapping.txt", 'r') as file:
    data = file.read()

# Split the data into lines and parse each line into a key-value pair
my_dict = {}
for num,line in enumerate(data.splitlines()):
    c = line.strip().split()[0]
    my_dict[num] = c

# Open a new file for writing
with open('image_net_map.json', 'w') as file:
    # Write the dictionary to the file in JSON format
    json.dump(my_dict, file)