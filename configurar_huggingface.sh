#!/usr/bin/env bash
# Configura el acceso al modelo gated de Pyannote (Community-1) en macOS.
# El token se introduce en el prompt seguro de Hugging Face.
# No debe escribirse dentro de un script, Markdown, archivo .env, comando compartido o chat.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ./.venv/bin/python ]; then
  echo "Falta .venv. Corre primero ./instalar.sh" >&2
  exit 1
fi

if ./.venv/bin/hf --help >/dev/null 2>&1; then
  ./.venv/bin/hf auth login
else
  ./.venv/bin/huggingface-cli login
fi

echo
echo "Ahora acepta las condiciones del modelo en:"
echo "https://huggingface.co/pyannote/speaker-diarization-community-1"
