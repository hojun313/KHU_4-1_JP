import sys
import numpy as np
import cv2
import os
import gs3drecon
import glob
from collections import defaultdict
import time

class MockGelSightDevice:
    def __init__(self, height, width):
        self.imgh = height
        self.imgw = width

def process_video(video_path, specific_output_dir, mmpp, net_path, gpu_mode, mask_markers, frame_height, frame_width):
    """하나의 영상 파일을 처리하여 이미지 시퀀스와 하이트 맵을 저장하는 함수"""
    print(f"\n--- {video_path} 처리 시작 ---")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"오류: 영상 파일을 열 수 없습니다 - {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"영상 해상도: {frame_width}x{frame_height}, FPS: {fps}")

    mock_dev = MockGelSightDevice(frame_height, frame_width)

    if gpu_mode:
        gpuorcpu = "cuda"
        print("GPU 사용 모드")
    else:
        gpuorcpu = "cpu"
        print("CPU 사용 모드")

    try:
        nn = gs3drecon.Reconstruction3D(mock_dev)
        net = nn.load_nn(net_path, gpuorcpu)
        print("신경망 모델 로드 완료")
    except Exception as e:
        print(f"오류: gs3drecon 초기화 실패: {e}")
        cap.release()
        return

    ret, f0 = cap.read()
    if not ret:
        print("오류: 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    roi = (0, 0, f0.shape[1], f0.shape[0])

    output_image_dir = os.path.join(specific_output_dir, 'images')
    output_heightmap_dir = os.path.join(specific_output_dir, 'heightmaps')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_heightmap_dir, exist_ok=True)

    frame_count = 0
    try:
        while True:
            ret, f1 = cap.read()
            if not ret:
                print("영상의 끝에 도달했습니다.")
                break

            frame_count += 1

            f1_cropped = f1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            # --- 검은 점 마스크 생성 ---
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([50, 50, 50])
            mask = cv2.inRange(f1_cropped, lower_black, upper_black)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

            # --- Inpainting 적용 ---
            inpainted_frame = cv2.inpaint(f1_cropped, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            output_image_path = os.path.join(output_image_dir, f'frame_{frame_count:05d}.png')
            cv2.imwrite(output_image_path, inpainted_frame)

            try:
                dm = nn.get_depthmap(inpainted_frame, mask_markers)

                output_heightmap_file = os.path.join(output_heightmap_dir, f'heightmap_{frame_count:05d}.tif')
                min_z = np.min(dm)
                max_z = np.max(dm)
                if max_z - min_z > 1e-6:
                    normalized_dm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imwrite(output_heightmap_file.replace('.tif', '.png'), normalized_dm)
                else:
                    flat_image = np.full(dm.shape, 128, dtype=np.uint8)
                    cv2.imwrite(output_heightmap_file.replace('.tif', '.png'), flat_image)

            except Exception as e:
                print(f"오류: 깊이 맵 계산 또는 저장 실패 (프레임 {frame_count}): {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user!')
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    finally:
        print(f"{video_path} 처리 완료.")
        cap.release()
        cv2.destroyAllWindows()

def main(argv):
    INPUT_ROOT_DIR = './allToHightMapVids'
    NET_FILE_PATH = 'nnmini.pt'
    MMPP = 0.0634
    GPU_MODE = True
    MASK_MARKERS_FLAG = False

    net_path = NET_FILE_PATH
    if not os.path.exists(net_path):
        print(f"오류: 신경망 모델 파일을 찾을 수 없습니다 - {net_path}")
        print("모델 파일(nnmini.pt)이 스크립트와 같은 경로에 있는지 확인하세요.")
        return

    all_video_files = []
    for item in os.listdir(INPUT_ROOT_DIR):
        subdir_path = os.path.join(INPUT_ROOT_DIR, item)
        if os.path.isdir(subdir_path):
            video_files_in_subdir = glob.glob(os.path.join(subdir_path, '*.avi')) + glob.glob(os.path.join(subdir_path, '*.mp4'))
            all_video_files.extend(video_files_in_subdir)

    if not all_video_files:
        print(f"경고: '{INPUT_ROOT_DIR}' 및 하위 폴더에 처리할 영상 파일이 없습니다.")
        return

    total_videos = len(all_video_files)
    processed_videos_count = 0
    start_time = time.time()

    print(f"\n--- 전체 영상 파일 수: {total_videos} ---")

    video_groups = defaultdict(list)
    for video_file in all_video_files:
        filename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(filename)[0]
        double_underscore_index = filename_without_ext.find('__')
        if double_underscore_index != -1:
            group_key_base = filename_without_ext[:double_underscore_index]
        else:
            group_key_base = filename_without_ext

        video_groups[group_key_base].append(video_file)

    for group_key_base, videos_in_group in video_groups.items():
        videos_in_group.sort()

        for i, video_file in enumerate(videos_in_group):
            processed_videos_count += 1

            output_folder_name = f"{group_key_base}_{i + 1}"
            parent_dir = os.path.dirname(video_file)
            specific_output_dir = os.path.join(parent_dir, 'output', output_folder_name)
            os.makedirs(specific_output_dir, exist_ok=True)
            os.makedirs(os.path.join(specific_output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(specific_output_dir, 'heightmaps'), exist_ok=True)

            elapsed_time = time.time() - start_time
            videos_per_second = processed_videos_count / elapsed_time if elapsed_time > 0 else 0
            remaining_videos = total_videos - processed_videos_count
            if videos_per_second > 0:
                estimated_remaining_time = remaining_videos / videos_per_second
                mins, secs = divmod(estimated_remaining_time, 60)
                print(f"전체 처리 중: 영상 {processed_videos_count}/{total_videos}, 남은 예측 시간: {int(mins)}분 {int(secs)}초")
            else:
                print(f"전체 처리 중: 영상 {processed_videos_count}/{total_videos}")

            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                process_video(video_file, specific_output_dir, MMPP, net_path, GPU_MODE, MASK_MARKERS_FLAG, frame_height, frame_width)
            else:
                print(f"오류: 영상 파일 '{video_file}'을 열 수 없습니다.")

    print("\n모든 영상 처리 완료.")

if __name__ == "__main__":
    main(sys.argv[1:])
