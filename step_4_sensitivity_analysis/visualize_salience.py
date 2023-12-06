import sys
import numpy as np
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency
import pickle

start = int(sys.argv[1])
block = int(sys.argv[2])
index = int(sys.argv[3])
vis_file = sys.argv[4]
data_file = sys.argv[5]
model_file = sys.argv[6]

end=start+block

print(start, end)

vis_sal = pickle.load( open(vis_file, "rb" ) )
model = load_model(model_file)

test_samples = pickle.load( open(data_file, "rb" ) )
layer_idx = utils.find_layer_idx(model, 'aux_output')

for j in range(start, end):
    print(j)
    if j < vis_sal.shape[0]:
        vis_sal[j] = visualize_saliency(model=model, layer_idx=layer_idx, filter_indices=[index], seed_input=test_samples[j], backprop_modifier='guided')
if j < vis_sal.shape[0]:
    print(vis_sal[end-1])

pickle.dump(vis_sal,open(vis_file, "wb" ))
