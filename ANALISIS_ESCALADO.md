# Análisis de capacidad y escalado de workers GPU

Fecha: 2026-10-20. Datos de demanda: semana del 12 al 18 de octubre de 2026
(`escuchas-simultaneas-168-horas-2026-10-12-al-18.csv`, hora de Lima).

## Resumen

- Una RTX 4070 Super procesa **~48 horas de audio por hora** (RTF ≈ 48),
  transcripción + diarización.
- La demanda de la semana fue de **~4,650 horas de audio**, con picos de 140
  escuchas simultáneas los días hábiles a las 7:00.
- **Nunca hacen falta más de 3 máquinas**, y la mayor parte del día basta con 1.
- Escalar según la hora cuesta **$37.27/semana (~$162/mes)**, casi lo mismo que una
  máquina fija 24/7 ($35.33/semana), y elimina el retraso de más de 5 h que
  tendría esa máquina fija los días hábiles.

## Rendimiento medido (RTF)

Instancia Vast.ai con 1× RTX 4070 Super, imagen `runpod-cu128`, Whisper
`large-v3` (`int8_float16`) y Pyannote `community-1`:

| | |
|---|---|
| Audio procesado | 48.22 h |
| Tiempo de reloj | 1 h 0 min 51 s (3,651 s) |
| RTF bruto | 48.22 × 3600 / 3651 = **47.5** |
| Trabajos reprocesados por fallas de red | 2 (un corte SSL al subir y un lease perdido) |

Sin esos dos reprocesos el RTF queda en **~48**, que es el valor usado en este
análisis. Las fallas de red se corrigieron con reintentos en `/complete` y
`/fail` y con un heartbeat cada 18 s que reintenta a los 5 s tras un fallo.

## Demanda

Supuesto: cada escucha simultánea genera una hora de audio por hora. Como el CSV
da el **máximo** de cada hora, la demanda real es igual o menor: el análisis es
conservador.

| | Lunes a viernes | Sábado | Domingo |
|---|---|---|---|
| Audio por día | ~865 h | 223 h | 101 h |
| Pico | 136–140 (7:00–8:00) | 44 | 16 |
| Horas con actividad | 4:00–24:00 | 5:00–23:00 | 5:00–23:00 |

Perfil de un día hábil (lunes):

| Hora | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Escuchas | 1 | 23 | 93 | 136 | 118 | 59 | 29 | 24 | 73 | 85 | 35 | 13 | 9 | 25 | 38 | 40 | 31 | 14 | 4 | 1 |
| Máquinas | 1 | 1 | 2 | 3 | 3 | 2 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Máquinas por hora = ⌈escuchas / 48⌉.

## Cálculo

Máquinas necesarias en cada momento:

```
N = ceil( (λ + B / D) / 48 )

λ = horas de audio que llegan por hora (escuchas simultáneas)
B = horas de audio pendientes en la cola
D = retraso máximo aceptable (horas); objetivo: ~1 h
```

## Costos por estrategia

Supuestos: $0.210/h por máquina, RTF 48, 15 min de arranque pagado sin producir
por cada máquina que se enciende (descarga de la imagen y carga de modelos) y
mes de 4.33 semanas. Simulación hora a hora sobre la semana del CSV.

| Estrategia | Horas-máquina/semana | Arranques | Costo/semana | Costo/mes | Backlog máximo | Retraso máximo |
|---|---|---|---|---|---|---|
| 1 máquina fija 24/7 | 168 | 1 | $35.33 | $153.11 | 261 h | **5.4 h** ❌ |
| 3 máquinas fijas 24/7 | 504 | 3 | $106.00 | $459.32 | 0 | 0 |
| **Por hora según demanda** | 172 | 22 | **$37.27** | **$161.53** | 0 | **~0** ✅ |
| Mínimo para retraso ≤ 1 h | 157 | 22 | $34.12 | $147.88 | 91 h | 59 min |

- Escalar según la demanda cuesta solo ~$2/semana más que una máquina fija y
  ahorra ~$69/semana (~$300/mes) frente a 3 máquinas fijas.
- Costo por hora de audio procesada: **~$0.008**.
- Los 22 arranques semanales cuestan ~$1.16/semana; el costo lo domina el tiempo
  de máquina.
- La estrategia "mínimo para retraso ≤ 1 h" ahorra $3/semana a cambio de
  acumular hasta 91 h de backlog los días hábiles por la mañana. No compensa.

## Patrón de escalado sugerido

Escalar según la hora cuesta casi lo mismo que una máquina fija y elimina el
retraso. El patrón entre semana sería:

| Horario (Lima) | Máquinas |
|---|---|
| 0:00–4:00 | 0 |
| 4:00–6:00 | 1 |
| 6:00–7:00 | 2 |
| 7:00–9:00 | 3 |
| 9:00–10:00 | 2 |
| 10:00–12:00 | 1 |
| 12:00–14:00 | 2 |
| 14:00–24:00 | 1 |

Los fines de semana basta con 1 máquina (sábado y domingo de 5:00 a 23:00).

## Propuesta: orquestador

Un `orquestador.py` que, por ahora, corre en una PC local y más adelante en un
servidor. No requiere cambios en el backend.

1. **Escalado por horario.** Usa el perfil semanal del CSV para calcular cuántas
   máquinas debería haber en cada momento y las lanza en Vast.ai unos 20 min
   antes de que se necesiten, para compensar el arranque. Reutiliza la lógica de
   `vast_launcher/lanzar_vast.py`.
2. **Apagado sin perder trabajos.** El worker gana una opción `--exit-when-idle N`:
   tras N minutos sin trabajos termina solo. El orquestador nunca destruye un
   worker ocupado; solo elimina en Vast las instancias que ya terminaron.
3. **Límites y alternativas.** Máximo de máquinas simultáneas y de $/h.
   Preferencia por hosts en América, cerca de `radio.datadaf.com`, para evitar
   los cortes de red observados con un host en Corea. Si no hay 4070 Super
   disponible, usa otras GPUs y ajusta la cuenta según su RTF.
4. **Paso a escalado reactivo.** Si el backend expone un endpoint con el backlog
   real (por ejemplo `GET /api/internal/jobs/stats` con trabajos pendientes, horas
   de audio pendientes y antigüedad del más viejo), el orquestador pasa a usar la
   fórmula de N con datos reales en lugar del horario.

## Limitaciones

- **Una sola semana de datos.** Feriados, eventos o semanas atípicas no están
  representados. El escalado por horario no reacciona a ellos hasta que exista
  el endpoint de backlog.
- **El CSV da máximos por hora**, no promedios: la demanda está sobrestimada, lo
  que da margen pero puede encender máquinas de más.
- **Los audios llegan en bloques de hasta ~90 min**, así que el retraso real que
  ve el usuario incluye la duración de la grabación además del procesamiento.
- **El RTF depende del host.** Se midió en una sola instancia; hosts con peor red
  o disco pueden rendir menos.
- **Disponibilidad en Vast.ai.** Que haya 4070 Super libres a ese precio en las
  horas pico no está garantizado.
- **Con el orquestador en una PC local**, si la PC está apagada no se enciende
  ninguna máquina.
