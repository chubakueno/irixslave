#!/usr/bin/env bash
# Instalador para macOS (Apple Silicon). Crea .venv e instala dependencias.
# No descarga los pesos de Whisper ni Pyannote todavía.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ] && command -v uv >/dev/null 2>&1; then
  # Preferido: el Python 3.12 autocontenido de "uv". El de Homebrew puede
  # traer un pyexpat roto contra el libexpat del sistema en versiones muy
  # nuevas/beta de macOS (falla ensurepip con un símbolo faltante).
  uv python install 3.12 >/dev/null 2>&1 || true
  PYTHON_BIN="$(uv python find 3.12 2>/dev/null || true)"
fi

if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3.12"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "No se encontró $PYTHON_BIN." >&2
  echo "Instálalo con: brew install uv   (o) brew install python@3.12" >&2
  exit 1
fi

echo "Usando $("$PYTHON_BIN" --version) ($PYTHON_BIN)"
"$PYTHON_BIN" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo
echo "Listo. Entorno creado en .venv/"
echo "Siguiente paso: ./configurar_huggingface.sh"
