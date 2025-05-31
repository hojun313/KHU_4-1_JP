#!/bin/bash

#SBATCH -J V2HeightmapTrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/hojun313/JP/slurm_logs/slurm-%A.out
#SBATCH -e /data/hojun313/JP/slurm_logs/slurm-%A.err

# --- 초기 환경 설정 ---
echo "---------------------------------------------------------------------"
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Directory: /data/hojun313/JP/"
mkdir -p /data/hojun313/JP/slurm_logs
echo "Slurm logs will be saved to /data/hojun313/JP/slurm_logs/"
echo "---------------------------------------------------------------------"

MODEL_TAG_ARG=$1
if [ -z "$MODEL_TAG_ARG" ]; then
    echo "오류: 모델 태그가 첫 번째 인자로 전달되지 않았습니다. (예: sbatch run_training.sh my_first_run)" >&2
    exit 1
fi
echo "Model Tag for this run: $MODEL_TAG_ARG"
echo "---------------------------------------------------------------------"

# --- 사용자 개인 Anaconda 환경 활성화 ---
CONDA_ENV_NAME="JP_3rdTry_GPU"
echo "Activating Anaconda environment: $CONDA_ENV_NAME"
source /data/hojun313/anaconda3/bin/activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'." >&2
    exit 1
fi
echo "Conda environment '$CONDA_DEFAULT_ENV' activated successfully."
echo "Python executable: $(which python)"
echo "---------------------------------------------------------------------"

# --- 데이터셋 준비 ---
NAS_TARBALL_PATH="/data/hojun313/JP/heightmap_dataset_83.tar"
LOCAL_SCRATCH_PARENT_DIR="/local_datasets/hojun313"
LOCAL_SCRATCH_BASE="${LOCAL_SCRATCH_PARENT_DIR}/${MODEL_TAG_ARG}_${SLURM_JOB_ID}"
LOCAL_DATA_FOR_PYTHON_SCRIPT="${LOCAL_SCRATCH_BASE}/heightmap_dataset"

# --- !!! 자동 공간 정리 로직 (CLEANUP_TARGET_PARENT_DIR 아래 모든 내용 삭제) 시작 !!! ---
echo "--- Cleaning up old job directories in $LOCAL_SCRATCH_PARENT_DIR ---"
echo "--- Cleanup logic finished ---"

echo "Preparing dataset for current job..."
mkdir -p "$LOCAL_SCRATCH_BASE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create current job's local scratch directory '$LOCAL_SCRATCH_BASE'." >&2
    exit 1
fi

echo "Copying dataset tarball from $NAS_TARBALL_PATH to local scratch..."
cp "$NAS_TARBALL_PATH" "${LOCAL_SCRATCH_BASE}/"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy dataset tarball." >&2; rm -rf "$LOCAL_SCRATCH_BASE"; exit 1;
fi

echo "Extracting dataset tarball..."
tar -xf "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" -C "$LOCAL_SCRATCH_BASE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract dataset tarball." >&2; rm -rf "$LOCAL_SCRATCH_BASE"; exit 1;
fi

echo "Removing local tarball..."
rm "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
echo "---------------------------------------------------------------------"
# --- 자동 공간 정리 로직 끝 ---

# --- 학습 스크립트 실행 ---
echo "Changing directory to $PROJECT_DIR"
cd "$PROJECT_DIR"
echo "Current working directory: $(pwd)"

echo "Starting Python training script (re_train.py)..."
python re_train.py \
    --data_root "$LOCAL_DATA_FOR_PYTHON_SCRIPT/heightmap_dataset" \
    --output_dir "/data/hojun313/JP/training_outputs" \
    --model_tag "$MODEL_TAG_ARG" \
    --batch_size 32 \
    --num_workers 8 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --encoder "resnet34" \
    --save_interval 5 \
    --image_size 256 \
    # --load_model "/data/hojun313/JP/training_outputs/PREVIOUS_TAG/checkpoints/checkpoint_epoch_10.pth.tar" \

PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script (re_train.py) exited with code $PYTHON_EXIT_CODE." >&2
fi
echo "---------------------------------------------------------------------"

# --- 로컬 데이터셋 정리 ---
echo "Final cleanup of local dataset directory: $LOCAL_SCRATCH_BASE"
rm -rf "$LOCAL_SCRATCH_BASE"
echo "Local dataset cleaned up successfully."
echo "---------------------------------------------------------------------"

echo "Job finished at $(date) with exit code $PYTHON_EXIT_CODE."
exit $PYTHON_EXIT_CODE