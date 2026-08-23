#!/usr/bin/env bash
# Uso: ./ejecutar.sh <audio_o_carpeta> [carpeta_salida]
# Corre en CPU (Apple Silicon no tiene backend CUDA para faster-whisper/pyannote).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "Uso: ./ejecutar.sh <ruta_audio_o_carpeta> [carpeta_salida]" >&2
  exit 1
fi

INPUT="$1"
shift
OUTPUT="resultados"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  OUTPUT="$1"
  shift
fi

if [ ! -x ./.venv/bin/python ]; then
  echo "Falta .venv. Corre primero ./instalar.sh" >&2
  exit 1
fi

./.venv/bin/python pipeline_transcripcion_diarizada.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --device cpu \
  --language es \
  "$@"
