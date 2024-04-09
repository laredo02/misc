
import torch
import tensorflow as tf

# PyTorch
if torch.cuda.is_available():
    pytorch_gpu_count = torch.cuda.device_count()
    pytorch_gpus = [torch.cuda.get_device_name(i) for i in range(pytorch_gpu_count)]
else:
    pytorch_gpu_count = 0
    pytorch_gpus = []


# TensorFlow
tf_gpu_devices = tf.config.list_physical_devices('GPU')
tensorflow_gpu_count = len(tf_gpu_devices)
tensorflow_gpus = [device.name for device in tf_gpu_devices]

print("\n\nPytorch: ", pytorch_gpu_count, pytorch_gpus, "\nTensorflow: ", tensorflow_gpu_count, tensorflow_gpus)


