import os
import numpy as np
import tensorflow as tf
from PIL import Image
## Crop the Image 
from PIL import ImageOps

from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [69, 69, 69]
  colormap[1] = [0, 255, 0]
  colormap[2] = [255, 0, 0]
  colormap[3] = [0, 0, 255]
  colormap[4] = [0, 255, 255]
  colormap[5] = [255, 0, 255]



  # colormap[6] = [0, 87, 150]
  # colormap[7] = [150, 0, 87]
  # colormap[8] = [87, 0, 255]

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]




# Set the path to the directory containing the .pb files
MODEL_PATH = "pb/mnv3Large_7/"

TFLITE_PATH = "tflite/mnv3Large_7/"

IMAGE_PATH = "images/mnv3Large_7/"

# Iterate over all files in the MODEL_PATH directory
for file_name in os.listdir(MODEL_PATH):
    if file_name.endswith(".pb"):
        # Load the TensorFlow model
        model_file = os.path.join(MODEL_PATH, file_name)
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file=model_file,
            input_arrays=['sub_2'],
            output_arrays=['ResizeBilinear_2'],
            input_shapes={'sub_2': [1, 257, 257, 3]}
        )

        # Convert to TFLite Model
        tflite_model = converter.convert()

        # Set the path to save the TFLite model
        tflite_path = os.path.join(TFLITE_PATH, file_name.replace(".pb", ".tflite"))

        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess the image
        image_path = "download.png"
        image = Image.open(image_path)
        input_size = tuple(input_details[0]['shape'][1:3])
        resized_image = image.resize(input_size, Image.BILINEAR)
        input_image = np.expand_dims(resized_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0

        # Run the inference
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        output_image = interpreter.get_tensor(output_details[0]['index'])
        output_image = np.squeeze(output_image)
        seg_map = np.argmax(output_image, axis=-1)

        # Overlay the segmentation on the input image
        seg_image = label_to_color_image(seg_map)
        seg_image = Image.fromarray(seg_image)
        overlay_image = Image.blend(resized_image, seg_image, alpha=0.7)

        # Save the overlayed image as a PNG file
        output_file_name = file_name.replace(".pb", ".png")
        output_path = os.path.join(IMAGE_PATH, output_file_name)
        overlay_image.save(output_path)
        print(f"Saved segmentation result: {output_path}")