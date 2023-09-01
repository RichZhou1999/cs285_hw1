import pickle
import torch
from torch import distributions
import numpy as np

file_path = 'expert_data_Ant-v4.pkl'

# Load data from the pickle file
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Now, 'loaded_data' contains the data that was saved in the pickle file
# print(loaded_data)

# mean = torch.tensor((1,2,3),dtype=torch.float32)
# logstd = torch.tensor((0,0,0), dtype=torch.float32)
# # mean = self.mean_net(observation)
# normal_dist = distributions.Normal(mean, logstd.exp())
# sample = normal_dist.sample((1,)).numpy()
# print(sample[0])

a = [{"a":1,"b":2},{"a":23,"b":3},{"a":33,"b":44}]
random_indices = np.random.permutation(3)
print(a[random_indices[:2]])