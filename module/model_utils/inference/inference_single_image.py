from ..load.load_onnxruntime import load_onnxruntime, predict_single_image

def show_image_with_acc(model_file, image): 
    model = load_onnxruntime(model_file)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # image = 'car.jpg'  # 예측할 이미지 경로
    predicted_class = predict_single_image(model, image, classes)
    # show_prediction(image, predicted_class, inference_time)

    return predicted_class
