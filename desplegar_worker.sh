#!/usr/bin/env bash
# Despliega este pipeline en una instancia Linux+CUDA ya creada (ej. vast.ai)
# y arranca el worker dentro de tmux, para que siga corriendo si te desconectas.
# El código viaja por rsync/scp sobre el SSH que ya usas para conectarte a mano
# -- no por "git clone" -- así el repo privado no necesita una deploy key nueva
# en cada instancia efímera.
#
# Uso: ./desplegar_worker.sh <host> <puerto> [--live] [--solo-instalar]
#   host, puerto: los del botón "Connect" de vast.ai (ssh -p <puerto> root@<host>)
#   --live: arranca el worker subiendo resultados reales. Sin esta bandera,
#           corre en dry-run (procesa y guarda local, no toca la cola de jobs).
#   --solo-instalar: copia el código e instala dependencias, pero NO arranca
#           el worker. Útil para dejar la instancia lista y correrla a mano.
#   --transcription-only / --diarization-only / --capabilities=<lista>: limita
#           qué salidas toma el worker de la cola (default: ambas).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
  echo "Uso: ./desplegar_worker.sh <host> <puerto> [--live] [--solo-instalar]" >&2
  exit 1
fi

HOST="$1"
PORT="$2"
LIVE_FLAG=""
WORKER_EXTRA=""
START_WORKER=1
shift 2
for arg in "$@"; do
  case "$arg" in
    --live) LIVE_FLAG="--live" ;;
    --solo-instalar|--no-arrancar) START_WORKER=0 ;;
    --transcription-only|--diarization-only) WORKER_EXTRA="$WORKER_EXTRA $arg" ;;
    --capabilities=*) WORKER_EXTRA="$WORKER_EXTRA $arg" ;;
    *) echo "Bandera desconocida: $arg" >&2; exit 1 ;;
  esac
done

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
# Sin --delete y con excludes explícitos. NO usar --filter=':- .gitignore' con
# --delete: si .venv/modelos no están en el árbol de origen, --delete los marca
# para borrar en el server y el filtro por-directorio no los protege a tiempo.
rsync -az \
  --exclude='.venv/' --exclude='.git/' --exclude='modelos/' \
  --exclude='resultados/' --exclude='resultados_worker/' \
  --exclude='muestra_datadaf/' --exclude='resultados_clasificador/' \
  --exclude='.worker_tmp/' --exclude='__pycache__/' --exclude='*.log' \
  --exclude='.env' \
  -e "ssh ${SSH_OPTS[*]}" \
  ./ "root@$HOST:$REMOTE_DIR/"

echo "Copiando .env..."
# Por el mismo canal SSH que ya usa rsync. No usamos scp acá: a scp el puerto
# se le pasa con -P mayúscula, y SSH_OPTS trae -p (minúscula = "preservar
# timestamps" en scp), lo que hacía que intentara conectar al puerto 22.
ssh "${SSH_OPTS[@]}" "root@$HOST" "cat > '$REMOTE_DIR/.env'" < .env

echo "Instalando dependencias en el servidor (rápido si ya estaban instaladas)..."
ssh "${SSH_OPTS[@]}" "root@$HOST" "cd $REMOTE_DIR && bash instalar_linux.sh"

if [ "$START_WORKER" -eq 0 ]; then
  echo
  echo "Listo (solo instalación). El worker NO se arrancó."
  echo "Para arrancarlo a mano:"
  echo "  ssh -p $PORT root@$HOST -t \"cd $REMOTE_DIR && tmux new-session -s worker './.venv/bin/python worker_transcripcion.py --persistent-models$WORKER_EXTRA $LIVE_FLAG'\""
  exit 0
fi

echo "Arrancando el worker dentro de tmux..."
ssh "${SSH_OPTS[@]}" "root@$HOST" \
  "cd $REMOTE_DIR && tmux new-session -d -s worker \"./.venv/bin/python worker_transcripcion.py --persistent-models$WORKER_EXTRA $LIVE_FLAG\""

echo
echo "Listo. Modo: $([ -n "$LIVE_FLAG" ] && echo LIVE || echo DRY-RUN)"
echo "Para ver el worker en vivo: ssh -p $PORT root@$HOST -t 'tmux attach -t worker'"
echo "(Ctrl+b luego d para salir del attach sin detener el worker)"
