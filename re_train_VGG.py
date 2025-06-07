import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import torchvision
import torchvision.models as models
import argparse
from torchvision import transforms
from re_dataset import TextureHeightmapDataset 
from re_model import create_model

class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layer_indices=[2, 7, 16, 25, 34], resize=True):
        super(VGGPerceptualLoss, self).__init__()
        print("VGGPerceptualLoss 초기화 중...")
        vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.features = nn.Sequential(*[vgg_features[i] for i in range(max(feature_layer_indices) + 1)])
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.feature_layer_indices = feature_layer_indices
        self.loss_fn = nn.L1Loss()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self.resize = resize

        print(f"VGG19 특징 추출 레이어 선택: {feature_layer_indices}")
        print("VGGPerceptualLoss 초기화 완료.")

    def _preprocess_image(self, img_tensor):
        img_tensor = (img_tensor + 1) / 2.0

        if img_tensor.size(1) == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
        if self.resize and (img_tensor.size(2) < 224 or img_tensor.size(3) < 224) :
            pass

        return self.normalize(img_tensor)

    def forward(self, pred_img, target_img):
        pred_img_vgg_input = self._preprocess_image(pred_img)
        target_img_vgg_input = self._preprocess_image(target_img)

        perceptual_loss = 0.0
        
        temp_pred = pred_img_vgg_input
        temp_target = target_img_vgg_input
        for i, layer in enumerate(self.features):
            temp_pred = layer(temp_pred)
            temp_target = layer(temp_target)
            if i in self.feature_layer_indices:
                perceptual_loss += self.loss_fn(temp_pred, temp_target)
                
        return perceptual_loss

def save_checkpoint(model, optimizer, scheduler, epoch, filename="my_checkpoint.pth.tar"):
    print("=> 체크포인트 저장")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, scheduler, lr_for_optimizer, device):
    print(f"=> 체크포인트 불러오기: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
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
        
    start_epoch = checkpoint.get("epoch", -1) + 1
    print(f"=> 체크포인트 로드 완료. 에포크 {start_epoch}부터 학습을 재개합니다.")
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_for_optimizer
        
    return start_epoch

def save_predictions_as_imgs(loader, model, epoch, folder, device, current_batch_size):
    model.eval()
    num_batches_to_save = 5 
    
    saved_count = 0
    for idx, (x, y) in enumerate(loader):
        if saved_count >= num_batches_to_save:
            break
            
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = model(x)
            preds = (preds * 0.5) + 0.5 
        
        y_unnormalized = (y * 0.5) + 0.5
        
        grid_tensor = torch.cat((y_unnormalized[:, :1, :, :], preds), dim=0)

        torchvision.utils.save_image(
            grid_tensor, 
            f"{folder}/comparison_epoch_{epoch+1}_batch_{idx}.png", 
            nrow=x.size(0)
        )
        saved_count += 1
    model.train()

def train_fn(loader, model, optimizer, l1_loss_fn, perceptual_loss_fn, lambda_perceptual, scaler, device):
    loop = tqdm(loader, desc="Training", leave=True)
    running_l1_loss = 0.0
    running_perceptual_loss = 0.0
    running_total_loss = 0.0
    num_samples = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets[:, :1, :, :].to(device=device)
        
        device_type = device.split(':')[0] if device != 'cpu' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=(device_type != 'cpu')):
            predictions = model(data)
            
            l1_loss = l1_loss_fn(predictions, targets)
            p_loss = perceptual_loss_fn(predictions, targets) 
            total_loss = l1_loss + (lambda_perceptual * p_loss) 
        
        optimizer.zero_grad()
        if device_type != 'cpu':
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        running_l1_loss += l1_loss.item() * data.size(0)
        running_perceptual_loss += p_loss.item() * data.size(0)
        running_total_loss += total_loss.item() * data.size(0)
        num_samples += data.size(0)
        
        loop.set_postfix(l1=l1_loss.item(), percep=p_loss.item(), total=total_loss.item())
    
    avg_l1_loss = running_l1_loss / num_samples if num_samples > 0 else 0.0
    avg_perceptual_loss = running_perceptual_loss / num_samples if num_samples > 0 else 0.0
    avg_total_loss = running_total_loss / num_samples if num_samples > 0 else 0.0
    
    print(f"에포크 평균 손실 - Total: {avg_total_loss:.4f}, L1: {avg_l1_loss:.4f}, Perceptual: {avg_perceptual_loss:.4f}")
    return avg_total_loss

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {DEVICE}")

    model = create_model(device=DEVICE, encoder_name=args.encoder)
    l1_loss_fn = nn.L1Loss().to(DEVICE)
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )

    start_epoch = 0
    if args.load_model:
        if os.path.exists(args.load_model):
            start_epoch = load_checkpoint(args.load_model, model, optimizer, scheduler, args.learning_rate, DEVICE)
        else:
            print(f"경고: 지정된 체크포인트 파일 '{args.load_model}'를 찾을 수 없습니다. 처음부터 학습합니다.")

    transform_texture = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_heightmap = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = TextureHeightmapDataset(
        data_root=args.data_root,
        transform_texture=transform_texture,
        transform_heightmap=transform_heightmap,
        exclude_heightmap_indices_up_to=args.exclude_indices
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    scaler_enabled = (DEVICE != 'cpu')
    scaler = torch.amp.GradScaler(DEVICE.split(':')[0] if DEVICE != 'cpu' else 'cpu', enabled=scaler_enabled)

    CHECKPOINT_DIR = os.path.join(args.output_dir, args.model_tag, "checkpoints")
    SAVE_PREDICTIONS_DIR = os.path.join(args.output_dir, args.model_tag, "saved_images")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- 에포크 {epoch+1}/{args.num_epochs} --- 학습률: {current_lr:.2e} ---")
        
        avg_epoch_loss = train_fn(loader, model, optimizer, l1_loss_fn, perceptual_loss_fn, args.lambda_perceptual, scaler, DEVICE)
        
        scheduler.step(avg_epoch_loss)
        
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
    
    parser.add_argument("--lr_patience", type=int, default=5, 
                        help="ReduceLROnPlateau의 patience 값 (손실 개선이 없는 것을 몇 에포크 참을지)")
    parser.add_argument("--lr_factor", type=float, default=0.1, 
                        help="ReduceLROnPlateau의 factor 값 (학습률 감소 비율)")
    
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="지각 손실의 가중치 (L1 손실에 대한 비율)")
    
    args, unknown = parser.parse_known_args()
    main(args)