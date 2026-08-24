#!/usr/bin/env bash
# Despliega este pipeline en una instancia Linux+CUDA ya creada (ej. vast.ai)
# y arranca el worker dentro de tmux, para que siga corriendo si te desconectas.
# El código viaja por rsync/scp sobre el SSH que ya usas para conectarte a mano
# -- no por "git clone" -- así el repo privado no necesita una deploy key nueva
# en cada instancia efímera.
#
# Uso: ./desplegar_worker.sh <host> <puerto> [--live]
#   host, puerto: los del botón "Connect" de vast.ai (ssh -p <puerto> root@<host>)
#   --live: arranca el worker subiendo resultados reales. Sin esta bandera,
#           corre en dry-run (procesa y guarda local, no toca la cola de jobs).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
  echo "Uso: ./desplegar_worker.sh <host> <puerto> [--live]" >&2
  exit 1
fi

HOST="$1"
PORT="$2"
LIVE_FLAG=""
if [ "${3:-}" = "--live" ]; then
  LIVE_FLAG="--live"
fi

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
REMOTE_DIR=/workspace/irixslave

if [ ! -f .env ]; then
  echo "Falta .env en este directorio (RADIO_API_TOKEN + HF_TOKEN)." >&2
  echo "Cópialo de .env.example y complétalo antes de desplegar." >&2
  exit 1
fi

echo "Esperando a que SSH responda en $HOST:$PORT..."
until ssh "${SSH_OPTS[@]}" "root@$HOST" true 2>/dev/null; do
  sleep 3
done

echo "Copiando código..."
rsync -az --delete \
  --exclude='.venv' --exclude='modelos' --exclude='resultados' \
  --exclude='resultados_worker' --exclude='.env' --exclude='.git' \
  --exclude='.worker_tmp' \
  -e "ssh ${SSH_OPTS[*]}" \
  ./ "root@$HOST:$REMOTE_DIR/"

echo "Copiando .env..."
scp "${SSH_OPTS[@]}" .env "root@$HOST:$REMOTE_DIR/.env"

echo "Instalando dependencias en el servidor (rápido si ya estaban instaladas)..."
ssh "${SSH_OPTS[@]}" "root@$HOST" "cd $REMOTE_DIR && bash instalar_linux.sh"

echo "Arrancando el worker dentro de tmux..."
ssh "${SSH_OPTS[@]}" "root@$HOST" \
  "cd $REMOTE_DIR && tmux new-session -d -s worker \"./.venv/bin/python worker_transcripcion.py --persistent-models $LIVE_FLAG\""

echo
echo "Listo. Modo: $([ -n "$LIVE_FLAG" ] && echo LIVE || echo DRY-RUN)"
echo "Para ver el worker en vivo: ssh -p $PORT root@$HOST -t 'tmux attach -t worker'"
echo "(Ctrl+b luego d para salir del attach sin detener el worker)"
