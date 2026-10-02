#!/bin/bash
# 等待 torch wheel 下载完成，然后安装 torch+torchvision+ultralytics 到系统 Python，并验证 CUDA。
set -e
cd "C:/Users/lenovo/Desktop/mathorcup"
PY="/c/Users/lenovo/AppData/Local/Programs/Python/Python312/python"
FILE="torch-2.11.0+cu128.whl"
TARGET=2753189216

echo "[install] waiting for torch download to finish (target=$TARGET bytes)..."
for i in $(seq 1 240); do
  sz=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$TARGET" ]; then
    echo "[install] download complete: $sz bytes"
    break
  fi
  if ! ps -p 708 >/dev/null 2>&1; then
    echo "[install] curl 708 gone; current size=$sz"
    if [ "$sz" -lt "$TARGET" ]; then
      echo "[install] WARNING: incomplete, resuming download with curl -C -"
      curl -sL -C - -o "$FILE" "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl"
      sz=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
      if [ "$sz" -ge "$TARGET" ]; then
        echo "[install] resume complete: $sz bytes"
        break
      fi
    fi
  fi
  sleep 15
done

sz=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
echo "[install] final torch size=$sz / $TARGET"
if [ "$sz" -lt "$TARGET" ]; then
  echo "[install] ERROR: torch download incomplete, aborting install"
  exit 1
fi

echo "[install] installing torch (this may take several minutes)..."
"$PY" -m pip install --no-deps "$FILE"

echo "[install] installing torchvision..."
"$PY" -m pip install --no-deps "torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl"

echo "[install] installing ultralytics + deps..."
"$PY" -m pip install ultralytics

echo "[install] verifying CUDA..."
"$PY" -c "import torch, torchvision, ultralytics; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "[install] DONE"
