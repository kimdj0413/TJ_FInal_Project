import tensorflow as tf

# 텐서플로우 버전 출력
print("TensorFlow Version:", tf.__version__)

# GPU 사용 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
    print("Number of GPUs:", len(gpus))
else:
    print("GPU is not available")

gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)