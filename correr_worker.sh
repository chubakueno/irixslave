#!/usr/bin/env bash
# Corre el worker en loop continuo y evita que la Mac se suspenda mientras esté activo.
# Uso: ./correr_worker.sh          (dry-run, no sube nada)
#      ./correr_worker.sh --live   (sube resultados reales)
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -x ./.venv/bin/python ]; then
  echo "Falta .venv. Corre primero ./instalar.sh" >&2
  exit 1
fi

echo "Worker corriendo con caffeinate (la Mac no se suspenderá mientras este proceso viva)."
echo "Presiona Ctrl+C para detener."

caffeinate -s ./.venv/bin/python worker_transcripcion.py "$@"
