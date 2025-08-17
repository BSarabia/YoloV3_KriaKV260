import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# Step 1: Load your Keras .h5 model
model = tf.keras.models.load_model('yolov3.h5')

# Step 2: Create the Vitis Inspector
inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B3136") #BS Annotation. Define the target DPU

# Step 3: Inspect the model with a specified input shape
inspector.inspect_model(
    model, 
    input_shape=[416, 416, 3],          # Specify the concrete input shape
    plot=True,                          # Generate a visual graph of the model
    plot_file="model.svg",              # Save the graph in an SVG file
    dump_results=True,                  # Dump inspection results
    dump_results_file="inspect_results.txt",  # Save results to a .txt file
    verbose=0                           # Set the level of detail in the output (0 for basic info)
)

print("Inspection completed. The results were saved in 'inspect_results.txt' and the graph in 'model.svg'.")

