#!/usr/bin/env bash
# Instalador para Linux + NVIDIA (ej. una instancia alquilada en vast.ai).
# Crea .venv e instala dependencias, incluyendo PyTorch con CUDA 13.0.
# No descarga los pesos de Whisper ni Pyannote todavía.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No se encontró $PYTHON_BIN." >&2
  echo "Instálalo (ej. 'apt install -y python3.12 python3.12-venv') o exporta PYTHON_BIN con la ruta correcta." >&2
  exit 1
fi

echo "Usando $("$PYTHON_BIN" --version) ($PYTHON_BIN)"
[ -d .venv ] || "$PYTHON_BIN" -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel

./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchaudio==2.11.0+cu130'

./.venv/bin/pip install -r requirements.txt

echo
echo 'Instalación terminada.'
./.venv/bin/python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA')"
echo
echo 'Siguiente paso: agrega HF_TOKEN y RADIO_API_TOKEN a tu .env.'
echo '(o corre ./configurar_huggingface.sh si prefieres "hf auth login" en vez de HF_TOKEN en .env)'
