from tensorflow.python import pywrap_tensorflow  
import os
model_dir = "checkpoint/perceptual_mse_1e-3xadv_5_6_15_00/"
checkpoint_path = os.path.join(model_dir, "srwgan")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
