import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision
import argparse
from torchvision import transforms

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
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {DEVICE}")

    model = create_model(device=DEVICE, encoder_name=args.encoder)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            start_epoch = load_checkpoint(args.load_model, model, optimizer, args.learning_rate)
        else:
            print(f"경고: 지정된 체크포인트 파일 '{args.load_model}'를 찾을 수 없습니다.")

    transform_texture = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_heightmap = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = TextureHeightmapDataset(
        data_root=DATA_ROOT,
        transform_texture=transform_texture,
        transform_heightmap=transform_heightmap
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    scaler = torch.amp.GradScaler(DEVICE.split(':')[0])

    CHECKPOINT_DIR = os.path.join(args.output_dir, args.model_tag, "checkpoints")
    SAVE_PREDICTIONS_DIR = os.path.join(args.output_dir, args.model_tag, "saved_images")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- 에포크 {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(loader, model, optimizer, loss_fn, scaler)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            print(f"\n--- 에포크 {epoch+1}, 저장 분기점 도달 ---")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(model, optimizer, epoch, filename=checkpoint_path)
            save_predictions_as_imgs(loader, model, epoch, folder=SAVE_PREDICTIONS_DIR, device=DEVICE)

if __name__ == "__main__":
    # ▼▼▼▼▼ 커맨드 라인 인자 파서 정의 ▼▼▼▼▼
    parser = argparse.ArgumentParser(description="Heightmap Generation Model Training")
    parser.add_argument("--data_root", type=str, required=True, help="데이터셋 루트 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="체크포인트와 결과 이미지가 저장될 상위 디렉토리")
    parser.add_argument("--model_tag", type=str, required=True, help="실험을 구분하기 위한 태그 (결과 저장 폴더 이름으로 사용)")
    parser.add_argument("--load_model", type=str, default=None, help="학습을 재개할 체크포인트 파일 경로")
    
    parser.add_argument("--encoder", type=str, default="resnet34", help="U-Net의 인코더 이름")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--num_epochs", type=int, default=25, help="총 학습 에포크 수")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader에서 사용할 CPU 워커 수")
    parser.add_argument("--save_interval", type=int, default=1, help="체크포인트를 저장할 에포크 주기")
    parser.add_argument("--image_size", type=int, default=256, help="이미지 리사이즈 크기")
    
    
    args, unknown = parser.parse_known_args()
    main(args)