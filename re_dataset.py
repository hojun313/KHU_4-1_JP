import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import shutil

class TextureHeightmapDataset(Dataset):
    def __init__(self, data_root, 
                 transform_texture=None, 
                 transform_heightmap=None, 
                 common_transform=None,
                 exclude_heightmap_indices_up_to=15):
        self.data_root = data_root
        self.transform_texture = transform_texture
        self.transform_heightmap = transform_heightmap
        self.common_transform = common_transform
        
        self.supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        
        self.excluded_heightmap_suffixes = []
        if exclude_heightmap_indices_up_to is not None and exclude_heightmap_indices_up_to >= 0:
            self.excluded_heightmap_suffixes = [f"_{i:05d}" for i in range(exclude_heightmap_indices_up_to + 1)]

        self.image_pairs = self._build_image_pairs()

        if not self.image_pairs:
            print(f"경고: '{self.data_root}' 경로에서 유효한 이미지 쌍을 찾을 수 없습니다. 데이터셋이 비어 있습니다.")
        else:
            print(f"총 {len(self.image_pairs)}개의 입력-하이트맵 쌍을 데이터셋으로 구축했습니다.")

    def _build_image_pairs(self):
        image_pairs = []
        
        items_in_data_root_pattern = os.path.join(self.data_root, '*')
        all_items_in_data_root = glob.glob(items_in_data_root_pattern)

        material_folders = []
        for item_path in all_items_in_data_root:
            is_dir = os.path.isdir(item_path)
            if is_dir:
                material_folders.append(item_path)

        if not material_folders:
            return []

        for material_folder_path in material_folders: 
            
            input_files_found_for_this_material = []
            for ext_idx, ext in enumerate(self.supported_extensions):
                input_files_pattern = os.path.join(material_folder_path, f'input_*{ext}')
                potential_input_files_for_ext = glob.glob(input_files_pattern)
                if potential_input_files_for_ext:
                    input_files_found_for_this_material.extend(potential_input_files_for_ext)

            if not input_files_found_for_this_material:
                continue
            
            input_file_path = input_files_found_for_this_material[0]
            
            output_base_folder = os.path.join(material_folder_path, "output")

            if not os.path.isdir(output_base_folder):
                continue

            condition_folder_pattern = os.path.join(output_base_folder, '*')
            all_items_in_output = glob.glob(condition_folder_pattern)
            
            condition_folders = []
            for item_path in all_items_in_output:
                is_dir = os.path.isdir(item_path)
                if is_dir:
                    condition_folders.append(item_path)
                
            for condition_folder_path in condition_folders:
                heightmaps_folder = os.path.join(condition_folder_path, "heightmaps")

                if not os.path.isdir(heightmaps_folder):
                    continue
                
                found_heightmap_files_for_condition = []
                for ext_idx, ext in enumerate(self.supported_extensions):
                    heightmap_files_pattern = os.path.join(heightmaps_folder, f'*{ext}')
                    potential_heightmap_files_for_ext = glob.glob(heightmap_files_pattern)
                    if potential_heightmap_files_for_ext:
                        found_heightmap_files_for_condition.extend(potential_heightmap_files_for_ext)
                

                for heightmap_file_path in found_heightmap_files_for_condition:
                    heightmap_filename_no_ext = os.path.splitext(os.path.basename(heightmap_file_path))[0]
                    
                    should_exclude = False
                    if self.excluded_heightmap_suffixes: 
                        for suffix in self.excluded_heightmap_suffixes:
                            if heightmap_filename_no_ext.endswith(suffix):
                                should_exclude = True
                                break
                    
                    if not should_exclude:
                        image_pairs.append((input_file_path, heightmap_file_path))
        
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, heightmap_path = self.image_pairs[idx]

        try:
            input_image = Image.open(input_path).convert("RGB")
            heightmap_image = Image.open(heightmap_path).convert("L") 
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - 입력: {input_path}, 하이트맵: {heightmap_path}. 오류: {e}")
            return None, None 
        except Exception as e:
            print(f"오류: 이미지 로드 중 문제 발생 (입력: {input_path}, 하이트맵: {heightmap_path}). 오류: {e}")
            return None, None

        original_input_size = input_image.size
        original_heightmap_size = heightmap_image.size

        if self.common_transform:
            try:
                input_image, heightmap_image = self.common_transform(input_image, heightmap_image)
            except Exception as e:
                print(f"오류: Common transform 적용 중 문제 발생. 입력: {input_path} (원본크기: {original_input_size}), 하이트맵: {heightmap_path} (원본크기: {original_heightmap_size}). 오류: {e}")

        if self.transform_texture:
            try:
                input_image = self.transform_texture(input_image)
            except Exception as e:
                print(f"오류: Texture transform 적용 중 문제 발생. 입력: {input_path} (Common Transform 후 크기: {input_image.size if isinstance(input_image, Image.Image) else '텐서'}). 오류: {e}")
                return None, None 
        
        if self.transform_heightmap:
            try:
                heightmap_image = self.transform_heightmap(heightmap_image)
            except Exception as e:
                print(f"오류: Heightmap transform 적용 중 문제 발생. 하이트맵: {heightmap_path} (Common Transform 후 크기: {heightmap_image.size if isinstance(heightmap_image, Image.Image) else '텐서'}). 오류: {e}")
                return None, None

        return input_image, heightmap_image

if __name__ == '__main__':

    your_actual_data_root = "heightmap_dataset"
    if not os.path.isdir(your_actual_data_root):
        print(f"오류: 지정된 데이터 경로 '{your_actual_data_root}'를 찾을 수 없습니다.")
        print("스크립트 내의 'your_actual_data_root' 변수를 올바른 경로로 수정해주세요.")
    else:
        print(f"지정된 데이터 경로에서 데이터셋 로딩 테스트를 시작합니다: '{your_actual_data_root}'")
        
        IMG_SIZE = 256
        texture_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        heightmap_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        transform_texture = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            texture_normalize 
        ])
        transform_heightmap = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            heightmap_normalize
        ])
        
        class SynchronizedRandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p
            def __call__(self, img1, img2):
                if random.random() < self.p:
                    return transforms.functional.hflip(img1), transforms.functional.hflip(img2)
                return img1, img2

        common_transform_instance = SynchronizedRandomHorizontalFlip(p=0.5)

        try:
            dataset = TextureHeightmapDataset(
                data_root=your_actual_data_root, 
                transform_texture=transform_texture,
                transform_heightmap=transform_heightmap,
                common_transform=common_transform_instance,
                exclude_heightmap_indices_up_to=5
            )

            print(f"Dataset length: {len(dataset)}")

            if len(dataset) > 0:
                def collate_fn_skip_none(batch):
                    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
                    if not batch: 
                        return torch.tensor([]), torch.tensor([]) 
                    return torch.utils.data.dataloader.default_collate(batch)

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_skip_none)
                
                print("\nDataLoader에서 첫 번째 배치 가져오기 시도:")
                try:
                    first_batch_retrieved = False
                    for i_batch, (texture_batch, heightmap_batch) in enumerate(dataloader):
                        if texture_batch.numel() == 0 and heightmap_batch.numel() == 0:
                            print("  배치에 유효한 데이터가 없습니다 (모든 샘플이 None으로 처리되었을 수 있음).")
                            continue
                        
                        print(f"  Batch {i_batch+1}:")
                        print(f"    Texture batch shape: {texture_batch.shape}, dtype: {texture_batch.dtype}")
                        print(f"    Heightmap batch shape: {heightmap_batch.shape}, dtype: {heightmap_batch.dtype}")
                        first_batch_retrieved = True
                        break
                    
                    if not first_batch_retrieved and len(dataset) > 0:
                        print("  DataLoader에서 유효한 배치를 가져올 수 없었습니다. 데이터셋 아이템 처리 중 오류가 많을 수 있습니다.")

                except Exception as e:
                    print(f"  DataLoader 테스트 중 오류: {e}")
            else:
                print("데이터셋이 비어있어 DataLoader 테스트를 진행할 수 없습니다.")

        except RuntimeError as e:
            print(f"데이터셋 생성 중 런타임 오류 발생: {e}")
        except Exception as e:
            print(f"예시 코드 실행 중 알 수 없는 오류 발생: {e}")