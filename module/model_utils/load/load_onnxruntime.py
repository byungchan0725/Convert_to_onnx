"""
onnx runtime 환경에서 모델 로드 
"""
import numpy as np 
import onnx
import onnxruntime as ort

# Data preprocessing 
from ..data_preprocessing.for_test import preprocess_image


def load_onnxruntime(onnx_path): 
    # onnx runtime에서 모델 로드 
    return ort.InferenceSession(onnx_path)

def predict_single_image(model, image, classes): 
    """
    Model: .onnx file 
    Image: image 
    Classes: image class
        - For example: classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        - CIFAR-10 
    """
    image = preprocess_image(image)

    inputs = {model.get_inputs()[0].name: image}
    outputs = model.run(None, inputs)
    pred = np.argmax(outputs[0], axis=1)

    return classes[pred[0]]
