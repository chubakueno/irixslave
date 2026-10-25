---
name: vast
description: Lista ofertas de Vast.ai que cumplen los requisitos del worker de transcripción y alquila instancias con vast_launcher/lanzar_vast.py. Úsalo cuando pidan ver máquinas/GPUs disponibles en Vast, precios, o lanzar/alquilar workers en Vast.ai.
---

# Vast.ai: listar ofertas y lanzar workers

Todo pasa por `vast_launcher/lanzar_vast.py` (stdlib, Python ≥ 3.9, sin
dependencias, sin GPU ni venv del worker). Ejecuta siempre desde la raíz del repo
y con `--env-file .env`, porque el `.env` está en la raíz, no junto al script.
Usa `python3` si `python` no existe, como en macOS.

Si la consulta falla con un error DNS (`socket.gaierror`, `nodename nor servname
provided`, `Name or service not known`) dentro de un sandbox con red restringida,
puede ser una restricción del entorno. En Codex, repite la consulta de solo
lectura con `sandbox_permissions="require_escalated"` y una `justification`
que solicite acceso a la API pública de Vast. No cambies credenciales ni
desactives TLS para resolver este error. Si también falla con acceso autorizado,
reporta el error sin dar por confirmada su causa. Esto no autoriza reintentar
un `--create` de resultado incierto; aplica la regla de creación indicada abajo.

Si falla con `CERTIFICATE_VERIFY_FAILED` en macOS, el Python instalado desde
python.org no tiene sus certificados. Pide al usuario que ejecute
`/Applications/Python 3.x/Install Certificates.command`.

## Reglas

- **Nunca leas, imprimas ni muestres `.env`**, ni pegues tokens en comandos. El
  script ya oculta los secretos en su salida.
- **Listar es gratis; `--create` cobra.** Antes de usar `--create`, muestra
  al usuario la oferta que se alquilaría (GPU, $/h, máquina, ubicación) y espera
  su confirmación explícita en este turno. Si piden N instancias, confirma N y el
  costo total por hora.
- **No reintentes un `--create` que falló por timeout o sin ID de instancia**
  ("Revisa la consola antes de repetir"): podría haber creado una instancia.
  Pide al usuario que revise https://cloud.vast.ai/instances/.
- Si no se indica la imagen (`--image`), pregúntala. No inventes una etiqueta.

## Listar ofertas

```bash
python3 vast_launcher/lanzar_vast.py --env-file .env --image <imagen>
```

Muestra la petición (secretos como `***`) y hasta 5 ofertas, ordenadas por
preferencia de GPU y luego por precio. No hace falta `VAST_API_KEY` para buscar.
Resume las ofertas en una tabla para el usuario (id, máquina, GPU, $/h, CUDA,
fiabilidad, ubicación).

Filtros útiles (combinables):

| Flag | Default | Para qué |
|---|---|---|
| `--gpu "RTX 4070S,RTX 4070"` | 4070S, 4070S Ti, 4070 Ti, 4070 | GPUs aceptadas, en orden de preferencia (nombres de Vast) |
| `--gpu-count N` | 1 | GPUs por instancia |
| `--max-price 0.3` | sin límite | USD/h total |
| `--min-reliability 0.95` | 0.95 | fiabilidad del host |
| `--min-download-mbps 200` | 200 | ancho de banda de bajada |
| `--disk-gb 70` | 70 | disco de la instancia |
| `--exclude-machine ID` | + `VAST_EXCLUDE_MACHINES` | evitar hosts problemáticos (repetible) |
| `--capabilities transcription` | ambas | solo Whisper; no requiere `HF_TOKEN` |

Siempre se exige host verificado, on-demand y CUDA ≥ 12.8.

## Lanzar

Tras la confirmación del usuario, usa los mismos flags que en el listado y agrega `--create`:

```bash
python3 vast_launcher/lanzar_vast.py --env-file .env --image <imagen> [filtros] --create
```

Alquila la primera oferta; si ya no está disponible prueba la siguiente (máx. 3).
Para alquilar exactamente la oferta que el usuario eligió, usa `--offer-id ID`
(sin buscar ni caer en otra).

Para **N instancias**, ejecuta el comando N veces, una a una y de forma secuencial. Detente al primer
error. Como cada alquiler retira su oferta del mercado, la siguiente ejecución
toma otra máquina. Si quieren máquinas distintas, pasa `--exclude-machine` con las
ya usadas.

Si falta configuración (`Falta configurar: ...`), dile al usuario qué variable
añadir a `.env` (`VAST_API_KEY`, `RADIO_API_TOKEN`, `HF_TOKEN`) sin pedirle el valor.

Al terminar, reporta los IDs de instancia, el $/h de cada una y recuerda que
siguen cobrando hasta que se destruyan en https://cloud.vast.ai/instances/ (no
hay apagado automático).
