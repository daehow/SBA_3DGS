#!/bin/bash
# ============================================================
# mipnerf360 데이터셋 다운로드
#
# 사용법:
#   bash env/mipnerf360.sh
#
# 약 12GB 다운로드 후 datasets/mipnerf360/ 에 압축 해제
# 씬: bicycle, bonsai, counter, garden, kitchen, room, stump
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${PROJECT_ROOT}/datasets"
ZIP_FILE="${DATASET_DIR}/360_v2.zip"
TARGET_DIR="${DATASET_DIR}/mipnerf360"

# 이미 있는지 확인
if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
    echo "mipnerf360 데이터셋이 이미 존재합니다: ${TARGET_DIR}"
    echo "씬: $(ls "$TARGET_DIR" | tr '\n' ' ')"
    exit 0
fi

mkdir -p "$DATASET_DIR"

# 다운로드
echo "=== mipnerf360 데이터셋 다운로드 (~12GB) ==="
echo "저장 경로: ${TARGET_DIR}"
echo ""
wget -c -O "$ZIP_FILE" http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

# 압축 해제
echo ""
echo "=== 압축 해제 중... ==="
unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

# zip 삭제
rm -f "$ZIP_FILE"
echo ""
echo "=== 완료 ==="
echo "경로: ${TARGET_DIR}"
echo "씬: $(ls "$TARGET_DIR" | tr '\n' ' ')"
du -sh "$TARGET_DIR"
