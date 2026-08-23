$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelFile = Join-Path $projectRoot 'modelos\large-v3-local\model.bin'
$progressFile = Join-Path $projectRoot 'transcripciones\_progreso.csv'
$expectedModelBytes = [int64]3087284237

if (Test-Path -LiteralPath $modelFile) {
    $bytes = (Get-Item -LiteralPath $modelFile).Length
    $percent = [math]::Round(100 * $bytes / $expectedModelBytes, 2)
    Write-Output "Modelo: $percent% ($bytes / $expectedModelBytes bytes)"
} else {
    Write-Output 'Modelo: 0%'
}

if (Test-Path -LiteralPath $progressFile) {
    $rows = Import-Csv -LiteralPath $progressFile
    $latest = $rows | Group-Object audio | ForEach-Object { $_.Group | Select-Object -Last 1 }
    $ok = @($latest | Where-Object status -eq 'ok').Count
    $errors = @($latest | Where-Object status -eq 'error').Count
    Write-Output "Transcripción: $ok / 216 completados; errores: $errors"
    $latest | Select-Object -Last 3 | Format-Table audio, status, seconds, duration -AutoSize
} else {
    Write-Output 'Transcripción: todavía no iniciada.'
}
