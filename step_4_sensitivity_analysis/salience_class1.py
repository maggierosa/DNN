import sys

start = sys.argv[1]
start = int(start)
block = int(sys.argv[2])
index = int(sys.argv[3])

end=start+block

print(start, end)

import numpy as np
from keras.models import load_model
vis_sal_class1_test = np.load('vis_sal_class1_test.dat')
model = load_model('model1.h5')
class1_test = np.load('class1_test.dat')
layer_idx=440
from vis.visualization import visualize_saliency

for j in range(start, end):
	if j<vis_sal_class1_test.shape[0]:
		vis_sal_class1_test[j] = visualize_saliency(model=model, layer_idx=layer_idx, filter_indices=[index], seed_input=class1_test[j], backprop_modifier='guided')

if j<vis_sal_class1_test.shape[0]:
	print(vis_sal_class1_test[end-1])

vis_sal_class1_test.dump('vis_sal_class1_test.dat')
