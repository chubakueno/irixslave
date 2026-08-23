param(
    [int]$Limit = 0,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$python = "$PSScriptRoot\.venv\Scripts\python.exe"
$arguments = @(
    "$PSScriptRoot\diarizar.py",
    '--audio-root', "$PSScriptRoot\audios",
    '--transcript-root', "$PSScriptRoot\transcripciones",
    '--output-root', "$PSScriptRoot\diarizaciones",
    '--cache-dir', "$PSScriptRoot\modelos\pyannote-cache",
    '--model', "$PSScriptRoot\modelos\pyannote-community-1",
    '--segmentation-batch-size', '6',
    '--embedding-batch-size', '16'
)
if ($Limit -gt 0) {
    $arguments += @('--limit', "$Limit")
}
if ($Force) {
    $arguments += '--force'
}

$env:MPLCONFIGDIR = "$PSScriptRoot\.cache\matplotlib"
$env:PYANNOTE_METRICS_ENABLED = 'false'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$PID | Set-Content -LiteralPath "$PSScriptRoot\diarizacion.pid"

& $python @arguments
$exitCode = $LASTEXITCODE
Remove-Item -LiteralPath "$PSScriptRoot\diarizacion.pid" -ErrorAction SilentlyContinue
exit $exitCode
