import os
import random
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to load 1000 random images from a directory and resize them to 416x416
def load_images(image_dir, num_images=1000, target_size=(416, 416), normalize=True):
    images = []
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    selected_files = image_files[:num_images]
    
    for img_file in selected_files:
        img_path = os.path.join(image_dir, img_file)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)

        if normalize:
            img_array = img_array / 255.0         
        
        images.append(img_array)
    
    # Convert to numpy array
    images = np.array(images)
    return images

# Load YOLOv3 model
model = tf.keras.models.load_model('yolov3.h5')

# Ensure the model has a fixed input shape (e.g., 416x416x3 for an image model)
cloned_model = tf.keras.models.clone_model(
    model,
    input_tensors=tf.keras.Input(shape=(416, 416, 3), name="image_input")
)
cloned_model.set_weights(model.get_weights())

# Prepare calibration dataset
calib_dataset = load_images(image_dir='images', num_images=200)

# Initialize VitisQuantizer
quantizer = vitis_quantize.VitisQuantizer(cloned_model, quantize_strategy='pof2s')

# Perform post-training quantization (PTQ) with 416x416 input size
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset,
    calib_batch_size=10,
    verbose=1,
    fold_conv_bn=True,                      # Fusionar BatchNorms a Conv (mejora velocidad)
    #convert_relu_to_relu6=False,           # Mantener ReLU (más eficiente que ReLU6 para DPU)
    convert_sigmoid_to_hard_sigmoid=True,  # Acelera sigmoid
    include_cle=True,                      # Cross-layer equalization mejora precisión
    cle_steps=1,
    include_fast_ft=False,                 # Desactivado para velocidad
    separate_conv_act=False,               # Fusión mejora eficiencia en inferencia
    weight_per_channel=False,              # Per-tensor es más rápido
    activation_per_channel=False,
    input_per_channel=False,
    bias_per_channel=False,
    weight_bit=8,
    activation_bit=8,
    bias_bit=8,
    input_bit=8
)

# Save the quantized model
quantized_model.save('yolov3_quantized.h5')

print("Quantized model saved as 'yolov3_quantized.h5'")

