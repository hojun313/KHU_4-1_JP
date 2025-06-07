import torch
import segmentation_models_pytorch as smp

def create_model(device, encoder_name="resnet34"):
    print(f"U-Net 모델을 생성합니다 (Encoder: {encoder_name}, Weights: ImageNet)")
    
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model.to(device)

if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = create_model(dev, encoder_name="mobilenet_v2") 
    dummy_input = torch.randn(2, 3, 256, 256).to(dev)
    output = test_model(dummy_input)

    print("\nsegmentation-models-pytorch U-Net 모델 테스트")
    print(f"입력 텐서 형태: {dummy_input.shape}")
    print(f"출력 텐서 형태: {output.shape}")
    assert dummy_input.shape[0] == output.shape[0]
    assert output.shape[1] == 1
    assert dummy_input.shape[2:] == output.shape[2:]
    print("테스트 통과!")