$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$modelDir = Join-Path $projectRoot 'modelos\large-v3-local'
$modelFile = Join-Path $modelDir 'model.bin'
$expectedModelBytes = [int64]3087284237
$modelUrl = 'https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin'

if (-not (Test-Path -LiteralPath $python)) {
    throw "No se encontró el entorno Python en $python"
}

New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

$currentBytes = 0
if (Test-Path -LiteralPath $modelFile) {
    $currentBytes = (Get-Item -LiteralPath $modelFile).Length
}

if ($currentBytes -ne $expectedModelBytes) {
    if ($currentBytes -gt $expectedModelBytes) {
        throw "model.bin tiene un tamaño inesperado: $currentBytes bytes"
    }
    Write-Output "Descargando large-v3: $currentBytes de $expectedModelBytes bytes ya disponibles."
    & curl.exe -L --fail --retry 20 --retry-all-errors --retry-delay 5 --connect-timeout 30 -C - -o $modelFile $modelUrl
    if ($LASTEXITCODE -ne 0) {
        throw "curl terminó con código $LASTEXITCODE"
    }
}

$downloadedBytes = (Get-Item -LiteralPath $modelFile).Length
if ($downloadedBytes -ne $expectedModelBytes) {
    throw "Descarga incompleta: $downloadedBytes de $expectedModelBytes bytes"
}

$snapshotRoot = Join-Path $projectRoot 'modelos\models--Systran--faster-whisper-large-v3\snapshots'
$snapshot = Get-ChildItem -LiteralPath $snapshotRoot -Directory | Select-Object -First 1
if (-not $snapshot) {
    throw "No se encontró el snapshot de configuración de large-v3"
}

@('config.json', 'preprocessor_config.json', 'tokenizer.json', 'vocabulary.json') | ForEach-Object {
    $source = Join-Path $snapshot.FullName $_
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Falta el archivo de modelo: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $modelDir $_) -Force
}

$env:PYTHONUNBUFFERED = '1'
$cudaBins = @(
    (Join-Path $projectRoot '.venv\Lib\site-packages\nvidia\cublas\bin'),
    (Join-Path $projectRoot '.venv\Lib\site-packages\nvidia\cudnn\bin'),
    (Join-Path $projectRoot '.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin')
)
foreach ($cudaBin in $cudaBins) {
    if (-not (Test-Path -LiteralPath $cudaBin)) {
        throw "Falta la biblioteca CUDA: $cudaBin"
    }
}
$env:PATH = ($cudaBins -join ';') + ';' + $env:PATH
Write-Output 'Modelo completo. Iniciando la transcripción de los 216 audios.'
& $python (Join-Path $projectRoot 'transcribir.py') `
    --input (Join-Path $projectRoot 'audios') `
    --output (Join-Path $projectRoot 'transcripciones') `
    --model $modelDir `
    --model-dir (Join-Path $projectRoot 'modelos') `
    --device cuda `
    --compute-type int8_float16 `
    --batch-size 12 `
    --beam-size 1 `
    --word-timestamps

if ($LASTEXITCODE -ne 0) {
    throw "La transcripción terminó con código $LASTEXITCODE"
}

Write-Output 'Proceso completo.'
