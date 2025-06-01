import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # 1. 스케줄러 임포트
from tqdm import tqdm
import os
import torchvision
# import torchvision.models as models
import argparse
from torchvision import transforms
import lpips

# 직접 만든 모듈 임포트 (re_dataset.py와 re_model.py가 같은 폴더에 있어야 합니다)
from re_dataset import TextureHeightmapDataset 
from re_model import create_model


# ----------------- 유틸리티 함수 -----------------
def save_checkpoint(model, optimizer, scheduler, epoch, filename="my_checkpoint.pth.tar"):
    """모델, 옵티마이저, 스케줄러 상태와 현재 에포크를 저장합니다."""
    print("=> 체크포인트 저장")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), # 스케줄러 상태 추가
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, scheduler, lr_for_optimizer, device):
    """체크포인트로부터 모델, 옵티마이저, 스케줄러 상태를 불러옵니다."""
    print(f"=> 체크포인트 불러오기: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device) # 현재 device로 로드
    
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("스케줄러 상태를 성공적으로 불러왔습니다.")
        except Exception as e:
            print(f"경고: 스케줄러 상태 로드 중 오류 발생 ({e}). 스케줄러는 초기 상태로 시작합니다.")
    else:
        print("경고: 체크포인트에 스케줄러 상태가 없거나 None입니다. 스케줄러는 초기 상태로 시작합니다.")
        
    start_epoch = checkpoint.get("epoch", -1) + 1 # .get()으로 이전 버전 호환성 확보
    print(f"=> 체크포인트 로드 완료. 에포크 {start_epoch}부터 학습을 재개합니다.")
    
    # 옵티마이저의 학습률을 현재 설정값으로 강제 업데이트 (선택 사항이나, 재개 시 명확성을 위해 권장)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_for_optimizer
        
    return start_epoch

def save_predictions_as_imgs(loader, model, epoch, folder, device, current_batch_size):
    """검증 데이터셋의 예측 결과를 [윗줄: 정답 / 아랫줄: 예측] 형태로 저장합니다."""
    model.eval()
    # 저장할 이미지 쌍의 최대 개수 (배치 단위)
    num_batches_to_save = 5 
    
    saved_count = 0
    for idx, (x, y) in enumerate(loader):
        if saved_count >= num_batches_to_save:
            break
            
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = model(x)
            # -1 ~ 1 범위를 0 ~ 1 범위로 변환
            preds = (preds * 0.5) + 0.5 
        
        # 정답(y)도 -1~1 범위이므로 동일하게 변환
        y_unnormalized = (y * 0.5) + 0.5
        
        # 윗줄: 정답, 아랫줄: 예측. 각 줄은 배치 크기만큼의 이미지를 가짐
        # y_unnormalized와 preds의 채널 수가 다를 수 있으므로, y_unnormalized도 1채널로 슬라이싱
        grid_tensor = torch.cat((y_unnormalized[:, :1, :, :], preds), dim=0)

        torchvision.utils.save_image(
            grid_tensor, 
            f"{folder}/comparison_epoch_{epoch+1}_batch_{idx}.png", 
            nrow=x.size(0) # 현재 배치의 실제 이미지 수 (마지막 배치는 더 작을 수 있음)
        )
        saved_count += 1
    model.train()

# ----------------- 메인 학습 함수 -----------------
def train_fn(loader, model, optimizer, l1_loss_fn, lpips_loss_fn, lambda_lpips, scaler, device):
    loop = tqdm(loader, desc="Training", leave=True)
    running_l1_loss = 0.0
    running_lpips_loss = 0.0
    running_total_loss = 0.0
    num_samples = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets[:, :1, :, :].to(device=device)
        
        device_type = device.split(':')[0] if device != 'cpu' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=(device_type != 'cpu')):
            predictions = model(data)
            
            l1_loss_val = l1_loss_fn(predictions, targets)
            # LPIPS는 이미지 쌍을 받아 스칼라 값을 반환, 보통 배치에 대한 평균을 내줌
            # LPIPS 입력은 [-1, 1] 범위의 이미지를 기대합니다. 현재 predictions와 targets가 이 범위이므로 바로 사용.
            lpips_loss_val = lpips_loss_fn(predictions, targets).mean() # 배치 평균 LPIPS 값
            total_loss = l1_loss_val + (lambda_lpips * lpips_loss_val) # 두 손실을 가중 합산
        
        optimizer.zero_grad()
        if device_type != 'cpu':
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        running_l1_loss += l1_loss_val.item() * data.size(0)
        running_lpips_loss += lpips_loss_val.item() * data.size(0) # .item()으로 스칼라 값 추출
        running_total_loss += total_loss.item() * data.size(0)
        num_samples += data.size(0)
        
        loop.set_postfix(l1=l1_loss_val.item(), lpips=lpips_loss_val.item(), total=total_loss.item())
    
    avg_l1_loss = running_l1_loss / num_samples if num_samples > 0 else 0.0
    avg_lpips_loss = running_lpips_loss / num_samples if num_samples > 0 else 0.0
    avg_total_loss = running_total_loss / num_samples if num_samples > 0 else 0.0
    
    print(f"에포크 평균 손실 - Total: {avg_total_loss:.4f}, L1: {avg_l1_loss:.4f}, LPIPS: {avg_lpips_loss:.4f}")
    return avg_total_loss 

# ----------------- 메인 실행 부분 -----------------
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {DEVICE}")

    model = create_model(device=DEVICE, encoder_name=args.encoder)
    l1_loss_fn = nn.L1Loss().to(DEVICE) # L1 손실 함수
    # ▼▼▼▼▼ 지각 손실 함수 초기화 ▼▼▼▼▼
    lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)

    for param in lpips_loss_fn.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 학습률 스케줄러 초기화
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',     # 손실 값이 최소화되는 것을 목표
        factor=args.lr_factor, # 학습률 감소 비율
        patience=args.lr_patience, # 이 에포크 수 동안 손실 개선이 없으면 학습률 감소
        verbose=True    # 학습률 변경 시 메시지 출력
    )

    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            start_epoch = load_checkpoint(args.load_model, model, optimizer, scheduler, args.learning_rate, DEVICE)
        else:
            print(f"경고: 지정된 체크포인트 파일 '{args.load_model}'를 찾을 수 없습니다. 처음부터 학습합니다.")

    # 이미지 변환 정의
    transform_texture = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # -1 ~ 1 범위로 정규화
    ])
    transform_heightmap = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # -1 ~ 1 범위로 정규화 (흑백)
    ])

    # 데이터셋 및 데이터로더 생성
    dataset = TextureHeightmapDataset(
        data_root=args.data_root,
        transform_texture=transform_texture,
        transform_heightmap=transform_heightmap,
        exclude_heightmap_indices_up_to=args.exclude_indices # 인자로부터 받도록 수정
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # GradScaler 초기화 (GPU 사용 시에만 활성화)
    scaler_enabled = (DEVICE != 'cpu')
    scaler = torch.amp.GradScaler(DEVICE.split(':')[0] if DEVICE != 'cpu' else 'cpu', enabled=scaler_enabled)

    # 결과 저장 디렉토리 (모델 태그 포함)
    CHECKPOINT_DIR = os.path.join(args.output_dir, args.model_tag, "checkpoints")
    SAVE_PREDICTIONS_DIR = os.path.join(args.output_dir, args.model_tag, "saved_images")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

    # 학습 루프
    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- 에포크 {epoch+1}/{args.num_epochs} --- 학습률: {current_lr:.2e} ---")
        
        avg_epoch_loss = train_fn(loader, model, optimizer, l1_loss_fn, lpips_loss_fn, args.lambda_lpips, scaler, DEVICE)
        
        # 에포크가 끝난 후 스케줄러 업데이트
        scheduler.step(avg_epoch_loss)
        
        # 체크포인트 및 예측 이미지 저장 (주기에 따라)
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            print(f"\n--- 에포크 {epoch+1}, 저장 분기점 도달 ---")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(model, optimizer, scheduler, epoch, filename=checkpoint_path)
            save_predictions_as_imgs(loader, model, epoch, folder=SAVE_PREDICTIONS_DIR, device=DEVICE, current_batch_size=args.batch_size)

if __name__ == "__main__":
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("="*60)
        print("segmentation_models_pytorch 라이브러리가 설치되지 않았습니다.")
        print("터미널에서 아래 명령어를 실행하여 설치해주세요:")
        print("pip install segmentation-models-pytorch")
        print("="*60)
        exit()

    parser = argparse.ArgumentParser(description="Heightmap Generation Model Training")
    parser.add_argument("--data_root", type=str, required=True, help="데이터셋 루트 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="./training_results", help="체크포인트와 결과 이미지가 저장될 상위 디렉토리")
    parser.add_argument("--model_tag", type=str, default="default_run", help="실험을 구분하기 위한 태그 (결과 저장 폴더 이름으로 사용)")
    parser.add_argument("--load_model", type=str, default=None, help="학습을 재개할 체크포인트 파일 경로")
    
    parser.add_argument("--encoder", type=str, default="resnet34", help="U-Net의 인코더 이름 (예: resnet34, efficientnet-b0)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="초기 학습률")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--num_epochs", type=int, default=25, help="총 학습 에포크 수")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader에서 사용할 CPU 워커 수")
    parser.add_argument("--save_interval", type=int, default=1, help="체크포인트를 저장할 에포크 주기")
    parser.add_argument("--image_size", type=int, default=256, help="이미지 리사이즈 크기")
    parser.add_argument("--exclude_indices", type=int, default=15, help="제외할 하이트맵 인덱스 상한값 (0부터 시작, 이 값까지 제외)")
    
    # 학습률 스케줄러 관련 인자
    parser.add_argument("--lr_patience", type=int, default=5, 
                        help="ReduceLROnPlateau의 patience 값 (손실 개선이 없는 것을 몇 에포크 참을지)")
    parser.add_argument("--lr_factor", type=float, default=0.1, 
                        help="ReduceLROnPlateau의 factor 값 (학습률 감소 비율)")
    
    parser.add_argument("--lambda_lpips", type=float, default=0.5, help="지각 손실의 가중치 (L1 손실에 대한 비율)")
    
    args, unknown = parser.parse_known_args() # 알 수 없는 인자는 무시
    main(args)