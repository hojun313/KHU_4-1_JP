import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision

from re_dataset import TextureHeightmapDataset 
from re_model import create_model

# ----------------- 하이퍼파라미터 및 설정 -----------------
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 1
DATA_ROOT = "heightmap_dataset"
CHECKPOINT_DIR = "checkpoints_smp"
SAVE_PREDICTIONS_DIR = "saved_images_smp"
ENCODER_NAME = "resnet34"
SAVE_EVERY_N_EPOCHS = 5
NUM_WORKERS = 8

# ----------------- 유틸리티 함수 (이전과 동일) -----------------
def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> 체크포인트 저장")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> 체크포인트 불러오기")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_predictions_as_imgs(loader, model, epoch, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx >= 5: break
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds * 0.5) + 0.5 
        
        y_unnormalized = (y * 0.5) + 0.5
        # 출력 채널(1)과 입력 채널(1)을 맞추기 위해 y도 1채널만 사용
        comparison = torch.cat([y_unnormalized[:, :1, :, :], preds], dim=3)

        torchvision.utils.save_image(comparison, f"{folder}/comparison_epoch_{epoch+1}_{idx}.png")
    model.train()

# ----------------- 메인 학습 함수 (이전과 동일) -----------------
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    mean_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets[:, :1, :, :].to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    print(f"에포크 평균 손실: {mean_loss/len(loader)}")

# ----------------- 메인 실행 부분 -----------------
def main():
    # ▼▼▼ 모델 초기화 부분 변경 ▼▼▼
    model = create_model(device=DEVICE, encoder_name=ENCODER_NAME)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    from torchvision import transforms
    IMG_SIZE = 256
    transform_texture = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_heightmap = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = TextureHeightmapDataset(
        data_root=DATA_ROOT,
        transform_texture=transform_texture,
        transform_heightmap=transform_heightmap
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- 에포크 {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(loader, model, optimizer, loss_fn, scaler)

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n--- 에포크 {epoch+1}, 저장 분기점 도달 ---")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(model, optimizer, epoch, filename=checkpoint_path)

            save_predictions_as_imgs(
                loader, model, epoch, folder=SAVE_PREDICTIONS_DIR, device=DEVICE
            )

if __name__ == "__main__":
    # 라이브러리 설치 확인
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("="*60)
        print("segmentation_models_pytorch 라이브러리가 설치되지 않았습니다.")
        print("터미널에서 아래 명령어를 실행하여 설치해주세요:")
        print("pip install segmentation-models-pytorch")
        print("="*60)
        exit()

    # 전역 변수 epoch를 save_predictions_as_imgs 함수에서 사용하기 위해 global 선언
    global epoch
    main()