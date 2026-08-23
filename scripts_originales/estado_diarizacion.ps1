$audioCount = @(Get-ChildItem -LiteralPath "$PSScriptRoot\audios" -Recurse -Filter '*.mp3').Count
$jsonCount = @(Get-ChildItem -LiteralPath "$PSScriptRoot\diarizaciones" -Recurse -Filter '*.diarization.json' -ErrorAction SilentlyContinue).Count
$speakerCount = @(Get-ChildItem -LiteralPath "$PSScriptRoot\diarizaciones" -Recurse -Filter '*.speakers.json' -ErrorAction SilentlyContinue).Count
$errors = 0
$progress = "$PSScriptRoot\diarizaciones\_progreso.csv"
if (Test-Path -LiteralPath $progress) {
    $rows = Import-Csv -LiteralPath $progress
    $latest = $rows | Group-Object audio | ForEach-Object { $_.Group | Select-Object -Last 1 }
    $errors = @($latest | Where-Object status -eq 'error').Count
}

[pscustomobject]@{
    audios = $audioCount
    diarizaciones = $jsonCount
    transcripciones_con_speakers = $speakerCount
    pendientes = [Math]::Max(0, $audioCount - $jsonCount)
    errores_actuales = $errors
} | Format-List
