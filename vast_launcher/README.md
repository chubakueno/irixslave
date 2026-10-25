# vast-launcher

Alquila en Vast.ai una instancia que arranca el worker de transcripción de
irixslave en modo **LIVE** con `--persistent-models`. Es un único script sin
dependencias: basta Python 3.9+.

La imagen se construye y publica desde irixslave (`Dockerfile.runpod`); este
proyecto solo la alquila.

## Origen del código

La fuente es la carpeta `vast_launcher/` del repo irixslave, para que el
launcher y la imagen cambien en el mismo commit. Edita siempre ahí. El repo
independiente se genera con:

```bash
git subtree split --prefix=vast_launcher -b vast-launcher
git push <remoto-vast-launcher> vast-launcher:main
```

Dentro de irixslave, el `.env` está en la raíz: ejecuta desde ahí con
`python vast_launcher/lanzar_vast.py --env-file .env ...`. En el repo
independiente, `.env` va junto al script y no hace falta `--env-file`.

## Uso

1. Copia `.env.example` a `.env` y completa `VAST_API_KEY`, `RADIO_API_TOKEN` y
   `HF_TOKEN`. Si la imagen es privada, añade `REGISTRY_USERNAME` y
   `REGISTRY_PASSWORD` (y `--registry-host` si no es Docker Hub).
2. Revisa las ofertas y luego alquila:

   ```bash
   python lanzar_vast.py --image tuusuario/irixslave:runpod-cu128
   python lanzar_vast.py --image tuusuario/irixslave:runpod-cu128 --create
   ```

   La primera orden muestra la petición con secretos ocultos y las cinco mejores
   ofertas (buscar es gratis y no requiere API key). `--create` alquila la primera; si Vast.ai
   responde que la oferta ya no está disponible, prueba la siguiente (máximo 3).
   Si la respuesta se pierde (timeout), no reintenta: revisa la consola antes de
   repetir para no pagar dos instancias. `--offer-id ID` alquila una oferta concreta.

Por defecto acepta, en este orden de preferencia, 1× RTX 4070S, RTX 4070S Ti
(16 GB), RTX 4070 Ti o RTX 4070; dentro de cada modelo elige la más barata. En la
web de Vast.ai la 4070S Ti aparece truncada como "RTX 4070S...". Además exige host verificado,
CUDA ≥ 12.8, fiabilidad ≥ 0.95, bajada ≥ 200 Mbps y 70 GB de disco. `--gpu` recibe
la lista en orden de preferencia con los nombres de Vast.ai, p. ej.
`--gpu "RTX 4070S,RTX 4070 Ti,RTX 4070"`. Ajusta también con `--gpu-count`,
`--max-price` (USD/h), `--min-reliability`, `--min-download-mbps` y `--disk-gb`.
`--exclude-machine ID` (repetible) o `VAST_EXCLUDE_MACHINES` evitan máquinas concretas.
Usa `--capabilities transcription` para cargar solo Whisper (no requiere `HF_TOKEN`).

Solo se envían a la instancia las variables del worker; `VAST_API_KEY` se queda
en la máquina que lanza. El worker ID se toma de `VAST_CONTAINERLABEL`, así que
cada instancia tiene uno propio.

Todo (modelos, resultados, cachés) vive en el disco de la instancia bajo
`/workspace`: sobrevive a detener la instancia pero se pierde al destruirla.
Una instancia detenida sigue cobrando almacenamiento. Los logs se ven en la
consola de Vast.ai o con `vastai logs ID`. No hay autoscaling ni apagado por
inactividad: destruye la instancia al terminar.

## Tests

```bash
python -m unittest -v
```

No hacen llamadas reales a la API.
