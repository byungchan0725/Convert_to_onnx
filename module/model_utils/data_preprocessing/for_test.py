"""
이 코드는 ResNet에 맞게 전처리가 되어 있습니다. 
- Model: ResNet 
    - input size: n x 3 x 244 x 244
- Dataset: CIFAR-10 
"""

from torchvision import transforms
from PIL import Image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet18의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10의 이미지 정규화
    ])
    image = Image.open(image_path).convert('RGB')  # 이미지 열기 및 RGB로 변환
    return transform(image).unsqueeze(0).numpy()  # 배치 차원 추가 및 numpy 배열로 변환
