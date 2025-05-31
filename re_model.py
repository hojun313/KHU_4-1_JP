import torch
import segmentation_models_pytorch as smp

def create_model(device, encoder_name="resnet34"):
    print(f"U-Net 모델을 생성합니다 (Encoder: {encoder_name}, Weights: ImageNet)")
    
    model = smp.Unet(
        encoder_name=encoder_name,      # 인코더 종류
        encoder_weights="imagenet",     # ImageNet으로 사전 학습된 가중치 사용
        in_channels=3,                  # 입력 채널 수 (RGB)
        classes=1,                      # 최종 출력 채널 수 (흑백 하이트맵)
    )
    return model.to(device)

if __name__ == '__main__':
    # 모델 테스트
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 'resnet34' 대신 다른 인코더로 테스트해볼 수도 있습니다.
    test_model = create_model(dev, encoder_name="mobilenet_v2") 
    dummy_input = torch.randn(2, 3, 256, 256).to(dev)
    output = test_model(dummy_input)

    print("\nsegmentation-models-pytorch U-Net 모델 테스트")
    print(f"입력 텐서 형태: {dummy_input.shape}")
    print(f"출력 텐서 형태: {output.shape}")
    assert dummy_input.shape[0] == output.shape[0]
    assert output.shape[1] == 1 # 출력 채널이 1인지 확인
    assert dummy_input.shape[2:] == output.shape[2:]
    print("테스트 통과!")