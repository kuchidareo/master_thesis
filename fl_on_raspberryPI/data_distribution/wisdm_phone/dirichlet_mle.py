import numpy as np
import json
import dirichlet

# Load the JSON data
with open('data_distribution/wisdm_phone/class_dist.json') as f:
    data = json.load(f)

# Convert to a 2D numpy array with shape [12, 40]
class_dist_array = np.zeros((12, 40))  # Initialize a 12x40 array

for entry in data:
    class_dist_array[entry['class'], entry['client']] = entry['value']  # Fill the array

print(class_dist_array)
print(dirichlet.mle(class_dist_array))