param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$OutputPath = "$PSScriptRoot\resultados",
    [int]$BeamSize = 1,
    [int]$BatchSize = 12,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Falta el entorno virtual. Ejecuta primero .\instalar.ps1'
}

$arguments = @(
    (Join-Path $PSScriptRoot 'pipeline_transcripcion_diarizada.py'),
    '--input', $InputPath,
    '--output', $OutputPath,
    '--whisper-model', 'large-v3',
    '--pyannote-model', 'pyannote/speaker-diarization-community-1',
    '--device', 'cuda',
    '--beam-size', "$BeamSize",
    '--batch-size', "$BatchSize"
)
if ($Force) {
    $arguments += '--force'
}

& $python @arguments
exit $LASTEXITCODE
