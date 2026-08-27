# Revisión del flujo TECO

Esta rama contiene únicamente código, configuración de ejemplo, pruebas y
documentación técnica. No incluye audios, transcripciones, tokens, IDs de jobs
ni datasets de validación.

## Cambio propuesto

- `WHISPER_BATCH_SIZE=0` es el perfil por defecto del worker en Windows/Linux.
- Whisper procesa secuencialmente cada audio para mantener continuidad entre
  bloques internos.
- La cola sigue siendo concurrente entre computadoras: leases, heartbeat,
  prefetch y modelos persistentes no cambian.
- VAD, Pyannote y el contrato del payload permanecen intactos.
- `WHISPER_BATCH_SIZE=12` restaura el perfil batch con reparación localizada.

## Evidencia agregada

- Gate dirigido: 33/33 fragmentos y 289/289 palabras de control.
- Regresión end-to-end: 20/20 audios y 47.199/47.199 palabras alineadas por
  Pyannote, con timestamps monótonos.
- Smoke persistente: 2.490 palabras en Whisper, diarización y payload.
- RTX 5060 Laptop: media estimada de 135,6 segundos por job completo; el
  control comparable aumentó aproximadamente 39% frente a batch+reparación.

Estas cifras provienen de una muestra dirigida y de una referencia generada
por el mismo modelo. No constituyen una garantía universal ni sustituyen una
escucha humana.

## Archivos principales para revisar

- `worker_transcripcion.py`: configuración, red, leases y selección del perfil.
- `scripts_originales/transcribir.py`: ruta secuencial y reversión batch.
- `scripts_originales/boundary_repair.py`: fallback de reparación batch.
- `scripts_originales/audio_compat.py`: decodificación FFmpeg reutilizable.
- `scripts_originales/diarizar.py`: carga de audio compatible para Pyannote.
- `02_analysis/tests/`: regresiones unitarias sin datos de producción.

## Comprobación local

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s 02_analysis\tests -v
```

El worker no debe iniciarse en modo `--live` durante la revisión. Para una
prueba local, usar un audio propio y credenciales fuera del repositorio.
