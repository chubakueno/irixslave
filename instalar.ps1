param(
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$venv = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3.12 -m venv $venv
    } else {
        $systemPython = Get-Command python.exe -ErrorAction Stop
        & $systemPython.Source -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo crear el entorno virtual con Python 3.12.'
    }
}

& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw 'Falló la actualización de pip.'
}

if ($CpuOnly) {
    & $python -m pip install torch==2.11.0 torchaudio==2.11.0
} else {
    & $python -m pip install --index-url https://download.pytorch.org/whl/cu130 `
        'torch==2.11.0+cu130' 'torchaudio==2.11.0+cu130'
}
if ($LASTEXITCODE -ne 0) {
    throw 'Falló la instalación de PyTorch.'
}

& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Falló la instalación de dependencias.'
}

Write-Host ''
Write-Host 'Instalación terminada.'
if (-not $CpuOnly) {
    & $python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA')"
}
Write-Host 'Siguiente paso: .\configurar_huggingface.ps1'
