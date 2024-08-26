import torch 
import torch.nn as nn
from torchvision import datasets, transforms, models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_file):
    # CIFAR-10 데이터셋을 예시로 
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    model.load_state_dict(torch.load(model_file)) 
    model.to(DEVICE)
    model.eval()
    return model

def export_to_onnx(model_file, file_name):
    # model_file: .pth 파일을 읽은 값 
    # file_name: 모델 파일 이름 
    model = load_model(model_file)

    # 예시로 사용하는 ResNet-18의 입력은 ( N x 3 x 224 x 224 ) 
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(DEVICE)
    
    # ONNX로 모델을 변환하여 저장
    # https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html 링크 참조 
    onnx_file_path = f"{file_name}.onnx"
    torch.onnx.export(
        model,               # 실행될 모델
        x,                   # 모델 입력값
        onnx_file_path,      # 모델 저장 경로
        export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지 여부
        opset_version=16,    # 모델을 변환할 때 사용할 ONNX 버전
        do_constant_folding=True,  # 최적화 시 상수폴딩을 사용할지 여부
        input_names=['input'],     # 모델의 입력값을 가리키는 이름
        output_names=['output'],   # 모델의 출력값을 가리키는 이름
        dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                      'output': {0: 'batch_size'}}
    )
    
    return onnx_file_path