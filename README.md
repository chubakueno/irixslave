# Pipeline replicable de transcripción diarizada

Este paquete reproduce el flujo utilizado para convertir audios en una transcripción con tiempos y etiquetas de hablante. Contiene solamente código y documentación.

**No contiene modelos, audios, transcripciones, cachés ni credenciales.** Los nombres de los modelos están configurados en los scripts y sus archivos se descargan únicamente cuando el usuario instala o ejecuta el pipeline.

## Modelos utilizados

- Transcripción: `large-v3`, mediante `faster-whisper`.
- Diarización: `pyannote/speaker-diarization-community-1`.

El flujo probado en el proyecto se ejecutó con una NVIDIA GeForce RTX 5060 Laptop GPU, CUDA, `int8_float16`, `batch_size=12`, `beam_size=1` y tiempos por palabra activados.

## Qué hace el programa

```text
Audio
  |
  +--> Faster-Whisper large-v3
  |      texto + inicio/final de cada palabra
  |
  +--> Pyannote Community-1
         intervalos SPEAKER_00, SPEAKER_01, ...
                  |
                  v
       alineación por coincidencia temporal
                  |
                  v
       JSON/TXT/SRT con hablante + texto + tiempos
```

Whisper reconoce las palabras. Pyannote analiza las voces, detecta cambios de hablante y agrupa fragmentos acústicamente similares. Después, `diarizar.py` asigna cada palabra de Whisper al intervalo de voz con mayor solapamiento temporal. Las palabras consecutivas del mismo hablante, separadas por no más de 0.8 segundos, se convierten en un turno.

## Advertencia sobre la identidad

`SPEAKER_00` no es un nombre personal y su alcance es local a cada archivo. Por ejemplo, `SPEAKER_00` de `parte_1.mp3` puede ser una persona diferente de `SPEAKER_00` de `parte_2.mp3`.

Para colocar nombres reales se necesita una etapa posterior de identificación de voz, utilizando audios de referencia y revisión humana. El script opcional `scripts_originales/generar_transcripcion_nombrada.py` aplica un mapa de identidades ya confirmado; no descubre identidades por sí mismo.

## Requisitos

- Windows 10 u 11.
- Python 3.12 de 64 bits.
- GPU NVIDIA y controlador compatible para la configuración CUDA recomendada.
- Espacio libre para descargar los modelos al ejecutar.
- Cuenta de Hugging Face para aceptar las condiciones de Pyannote Community-1.

El pipeline también admite CPU con parámetros manuales, pero la diarización y el modelo `large-v3` serán considerablemente más lentos.

## Instalación

Abra PowerShell dentro de esta carpeta:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\instalar.ps1
```

El instalador crea `.venv` y descarga las dependencias. No descarga todavía los pesos de Whisper ni Pyannote.

Después, acepte las condiciones del modelo en:

`https://huggingface.co/pyannote/speaker-diarization-community-1`

Configure la autorización:

```powershell
.\configurar_huggingface.ps1
```

El token se introduce en el prompt seguro de Hugging Face. No debe escribirse dentro de un script, Markdown, archivo `.env`, comando compartido o chat.

## Uso sencillo

Para un audio:

```powershell
.\ejecutar.ps1 -InputPath "C:\ruta\entrevista.mp3" -OutputPath "C:\ruta\resultado"
```

Para una carpeta completa y sus subcarpetas:

```powershell
.\ejecutar.ps1 -InputPath "C:\ruta\audios" -OutputPath "C:\ruta\resultado"
```

El primer uso descargará los archivos de los modelos. Esas descargas se almacenarán en `modelos/`, una carpeta excluida de este paquete.

## Uso directo de Python

```powershell
.\.venv\Scripts\python.exe .\pipeline_transcripcion_diarizada.py `
  --input "C:\ruta\audios" `
  --output "C:\ruta\resultado" `
  --whisper-model "large-v3" `
  --pyannote-model "pyannote/speaker-diarization-community-1" `
  --device cuda `
  --beam-size 1 `
  --batch-size 12
```

Si se conoce el número exacto de hablantes de un audio, se puede indicar:

```powershell
.\.venv\Scripts\python.exe .\pipeline_transcripcion_diarizada.py `
  --input "C:\ruta\entrevista.mp3" `
  --output "C:\ruta\resultado" `
  --num-speakers 2
```

No conviene fijar `--num-speakers` si el dato no es seguro. También existen `--min-speakers` y `--max-speakers`.

## Formatos admitidos

El orquestador admite MP3, WAV, M4A, FLAC, OGG, AAC, WMA y archivos MP4 con pista de audio. La disponibilidad concreta de códecs depende de PyAV/FFmpeg.

## Resultados

La carpeta elegida como salida contiene:

```text
resultado/
  transcripciones/
    audio.json              texto y tiempos de Whisper
    audio.txt
    audio.srt
  diarizaciones/
    audio.diarization.json  intervalos detectados
    audio.rttm              formato estándar de diarización
    audio.embeddings.npz    huellas vocales locales
    audio.speakers.json     transcripción diarizada principal
    audio.speakers.txt
    audio.speakers.srt
  indice_resultados.json
```

El archivo que normalmente debe consumirse es `audio.speakers.json`. Su estructura esencial es:

```json
{
  "speaker_scope": "local_to_audio_file",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "turns": [
    {
      "start": 10.42,
      "end": 18.73,
      "speaker": "SPEAKER_00",
      "text": "Texto pronunciado durante este turno."
    }
  ]
}
```

## Cómo se calcula la asignación de hablante

1. Whisper produce palabras con `start` y `end`.
2. Pyannote produce intervalos exclusivos por hablante.
3. Para cada palabra se calcula el solapamiento con esos intervalos.
4. Se elige el hablante con mayor solapamiento.
5. Si no existe solapamiento, se usa el intervalo más cercano solamente cuando está a 0.75 segundos o menos; de otro modo se asigna `UNKNOWN`.
6. Las palabras consecutivas del mismo hablante se agrupan en turnos.

La diarización exclusiva facilita asignar una sola etiqueta a cada palabra. Los intervalos regulares también se conservan para revisar casos de voces superpuestas.

## Calidad y límites

- Whisper no garantiza transcripción perfecta; ruido, música, nombres propios y mala dicción pueden producir errores.
- Pyannote puede separar incorrectamente voces muy parecidas, intervenciones muy breves o voces superpuestas.
- La música sin habla puede generar segmentos vacíos o falsos positivos ocasionales.
- `beam_size=1` prioriza velocidad. Para una revisión concreta se puede probar `beam_size=5`, con mayor tiempo de proceso y sin garantía de corregir todas las palabras.
- La normalización de nombres propios y lugares debe hacerse como etapa derivada y auditable, sin modificar los audios originales.

## Reanudación

Los scripts registran progreso y omiten resultados completos. Si el proceso se interrumpe, ejecute el mismo comando para continuar. Use `-Force` únicamente si desea recalcular resultados existentes.

## Scripts incluidos

### Entrada recomendada

- `pipeline_transcripcion_diarizada.py`: coordina las dos etapas y genera el índice final.
- `ejecutar.ps1`: acceso sencillo para Windows.
- `instalar.ps1`: crea el entorno e instala dependencias.
- `configurar_huggingface.ps1`: configura el acceso al modelo gated sin guardar el token en el proyecto.
- `verificar_paquete.py`: comprueba sintaxis, archivos obligatorios y ausencia de modelos, audios y tokens.

### Copias exactas del pipeline usado

La carpeta `scripts_originales/` conserva los productores, monitores y auditores usados en el proyecto. Los lanzadores originales reflejan la estructura de aquel proyecto y se incluyen como referencia; para una instalación nueva se debe utilizar `pipeline_transcripcion_diarizada.py`.

| Script original | Función en el flujo |
|---|---|
| `transcribir.py` | Ejecuta Faster-Whisper, conserva tiempos por palabra y genera JSON, TXT y SRT. |
| `diarizar.py` | Ejecuta Pyannote, guarda intervalos/embeddings y alinea las palabras con hablantes. |
| `ejecutar_transcripcion.ps1` | Lanzador exacto empleado para la transcripción del lote original. |
| `ejecutar_diarizacion.ps1` | Lanzador exacto empleado para la diarización del lote original. |
| `estado_transcripcion.ps1` | Consulta rápida del avance de Whisper. |
| `estado_diarizacion.ps1` | Consulta rápida del avance de Pyannote. |
| `auditar_calidad.py` | Audita cobertura, estructura y señales de baja confianza de Whisper. |
| `auditar_diarizacion.ps1` | Audita completitud, hablantes, errores, velocidad y alertas de diarización. |
| `generar_transcripcion_nombrada.py` | Sustituye etiquetas locales por nombres únicamente cuando existe un mapa confirmado. |

`configurar_huggingface.ps1` también es copia del script usado. Se coloca en la raíz porque sigue siendo el método recomendado para autenticar la instalación nueva.

## macOS (Apple Silicon)

Este paquete también corre en Mac (M1/M2/M3/M4). La transcripción usa `mlx-whisper` (motor `--transcription-engine mlx`, por defecto en `ejecutar.sh` y en el worker), que corre sobre Metal/GPU vía Apple MLX — mucho más rápido que CPU pura. El modelo por defecto es `mlx-community/whisper-large-v3-mlx` (mismos pesos que `large-v3`, sin cuantizar, sin la pérdida de calidad de una cuantización agresiva a 4/8 bit). Ojo: variantes como `-fp16` u `-8bit` de ese mismo repo empaquetan los pesos como `model.safetensors`, un nombre que el loader de `mlx_whisper==0.4.3` (la última versión publicada) no reconoce — solo busca `weights.npz` o `weights.safetensors`. Si cambias de modelo, verifica primero el nombre del archivo de pesos en la pestaña "Files" del repo en Hugging Face.

La diarización usa el motor `soniqo` (`--diarization-engine soniqo`, por defecto en el worker): llama al CLI [`speech`](https://github.com/soniqo/speech-swift) (`brew install speech`) con `--engine community1`, que corre el mismo modelo `pyannote/speaker-diarization-community-1` pero sobre CoreML/Neural Engine en vez de PyTorch/CPU — en pruebas con audio real, ~17× tiempo real (vs. CPU) con conteo y proporción de hablantes casi idénticos a `pyannote-audio`. `scripts_originales/diarizar_soniqo.py` hace de wrapper: convierte el audio a WAV, llama al CLI, y reutiliza la misma lógica de alineación palabra↔hablante de `diarizar.py`.

Para usar el pipeline 100% PyTorch original (útil para comparar, o si `speech` no está disponible) pasa `--diarization-engine pyannote` a `pipeline_transcripcion_diarizada.py`; esa ruta sigue corriendo en CPU (`--device cpu`), ya que pyannote-audio no tiene backend Metal/MPS maduro. Igual para transcripción: `--transcription-engine faster-whisper --whisper-model large-v3` vuelve al motor original CTranslate2/CPU.

Instalación:

```bash
brew install python@3.12
./instalar.sh   # también instala 'speech' (brew) si falta
./configurar_huggingface.sh   # acepta antes las condiciones en huggingface.co/pyannote/speaker-diarization-community-1
```

Uso directo:

```bash
./ejecutar.sh "/ruta/audio.mp3" resultado
```

## Worker de polling (radio.datadaf.com)

`worker_transcripcion.py` hace polling de trabajos en la API interna, descarga el audio, corre este mismo pipeline localmente y sube el resultado.

1. Copia `.env.example` a `.env` y completa `RADIO_API_TOKEN` (nunca lo pegues en el chat ni lo commitees; `.env` ya está en `.gitignore`).
2. Corre un solo job en modo de prueba (no sube nada, solo guarda el resultado en `resultados_worker/<job_id>.json` y lo imprime):
   ```bash
   ./.venv/bin/python worker_transcripcion.py --once
   ```
3. Revisa el JSON generado. Cuando estés conforme, sube resultados reales:
   ```bash
   ./.venv/bin/python worker_transcripcion.py --once --live      # un solo job
   ./.venv/bin/python worker_transcripcion.py --live             # loop continuo
   ```

Cada job renueva el lease con un heartbeat cada 30 s mientras se procesa (margen de seguridad frente al TTL del lease en el backend, para que no se rehabilite el job para otro worker). El campo `text` que se sube queda formateado como `[SPEAKER_00] texto...` por turno, y cada `word` incluye `text`, `start`, `end`, `type: "word"`, `speaker_id` y `logprob` (derivado de la probabilidad de Whisper).

## Verificación del paquete

Esta comprobación no descarga modelos ni procesa audio:

```powershell
python .\verificar_paquete.py
```

El resultado esperado es `PAQUETE_APROBADO`.
