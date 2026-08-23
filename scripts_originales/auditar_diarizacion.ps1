$ErrorActionPreference = 'Stop'

$audioRoot = Join-Path $PSScriptRoot 'audios'
$transcriptRoot = Join-Path $PSScriptRoot 'transcripciones'
$outputRoot = Join-Path $PSScriptRoot 'diarizaciones'
$progressPath = Join-Path $outputRoot '_progreso.csv'
$processIdPath = Join-Path $PSScriptRoot 'diarizacion.pid'

$audioCount = @(Get-ChildItem -LiteralPath $audioRoot -Recurse -Filter '*.mp3').Count
$diarizationFiles = @(Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter '*.diarization.json' -ErrorAction SilentlyContinue)
$speakerFiles = @(Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter '*.speakers.json' -ErrorAction SilentlyContinue)

$latestErrors = @()
if (Test-Path -LiteralPath $progressPath) {
    $latestRows = Import-Csv -LiteralPath $progressPath |
        Group-Object audio |
        ForEach-Object { $_.Group | Select-Object -Last 1 }
    $latestErrors = @($latestRows | Where-Object status -eq 'error')
}

$rows = foreach ($diarizationFile in $diarizationFiles) {
    $diarization = Get-Content -LiteralPath $diarizationFile.FullName -Raw | ConvertFrom-Json
    $speakerFilePath = $diarizationFile.FullName -replace '\.diarization\.json$', '.speakers.json'
    $units = @()
    $turns = @()
    if (Test-Path -LiteralPath $speakerFilePath) {
        $speakerData = Get-Content -LiteralPath $speakerFilePath -Raw | ConvertFrom-Json
        $units = @($speakerData.units)
        $turns = @($speakerData.turns)
    }

    $exclusive = @($diarization.exclusive_diarization)
    $speechSeconds = ($exclusive | ForEach-Object { [double]$_.end - [double]$_.start } | Measure-Object -Sum).Sum
    $durationSeconds = [double]$diarization.duration_seconds
    $coverage = if ($durationSeconds -gt 0) { $speechSeconds / $durationSeconds } else { 0 }
    $unassigned = @($units | Where-Object { [string]::IsNullOrWhiteSpace($_.speaker) }).Count
    $shortTurns = @($turns | Where-Object { ([double]$_.end - [double]$_.start) -lt 0.75 }).Count
    $shortTurnRatio = if ($turns.Count -gt 0) { $shortTurns / $turns.Count } else { 0 }

    $fragmentedSpeakers = @($turns |
        Where-Object speaker -ne 'UNKNOWN' |
        Group-Object speaker |
        ForEach-Object {
            $speakerSeconds = ($_.Group | ForEach-Object { [double]$_.end - [double]$_.start } | Measure-Object -Sum).Sum
            if ($_.Count -ge 10 -and $speakerSeconds -lt 30 -and ($speakerSeconds / $_.Count) -lt 1) {
                $_.Name
            }
        })

    $transcriptHasSpeech = $null
    $relativeTranscript = [IO.Path]::ChangeExtension([string]$diarization.audio, '.json')
    $transcriptPath = Join-Path $transcriptRoot $relativeTranscript
    if (Test-Path -LiteralPath $transcriptPath) {
        $transcript = Get-Content -LiteralPath $transcriptPath -Raw | ConvertFrom-Json
        $transcriptHasSpeech = @($transcript.segments).Count -gt 0 -or [double]$transcript.duration_after_vad -gt 0
    }

    $warnings = @()
    $notes = @()
    if ([int]$diarization.num_speakers -lt 1) { $warnings += 'sin_hablantes' }
    if ([int]$diarization.num_speakers -gt 12) { $warnings += 'demasiados_hablantes' }
    if ($units.Count -eq 0 -and $transcriptHasSpeech -eq $false) { $notes += 'sin_voz_segun_whisper' }
    elseif ($units.Count -eq 0) { $warnings += 'sin_texto_alineado' }
    if ($units.Count -gt 0 -and ($unassigned / $units.Count) -gt 0.01) { $warnings += 'texto_sin_hablante' }
    if ($coverage -lt 0.15) { $warnings += 'muy_poca_voz' }
    if ($fragmentedSpeakers.Count -gt 0) { $warnings += "speaker_fragmentado:$($fragmentedSpeakers -join '+')" }

    [pscustomobject]@{
        audio = $diarization.audio
        speakers = [int]$diarization.num_speakers
        duration_min = [math]::Round($durationSeconds / 60, 1)
        processing_s = [math]::Round([double]$diarization.processing_seconds, 1)
        speech_coverage_pct = [math]::Round($coverage * 100, 1)
        words = $units.Count
        unassigned_words = $unassigned
        turns = $turns.Count
        short_turn_pct = [math]::Round($shortTurnRatio * 100, 1)
        warnings = $warnings -join ','
        notes = $notes -join ','
    }
}

$warningRows = @($rows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.warnings) })
$noteRows = @($rows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.notes) })
$averageSeconds = if ($rows.Count -gt 0) {
    [math]::Round(($rows | Measure-Object -Property processing_s -Average).Average, 1)
} else { 0 }

[pscustomobject]@{
    audios = $audioCount
    completados = $diarizationFiles.Count
    pendientes = [math]::Max(0, $audioCount - $diarizationFiles.Count)
    errores = $latestErrors.Count
    advertencias_calidad = $warningRows.Count
    segundos_promedio_por_archivo = $averageSeconds
} | Format-List

if ($warningRows.Count -gt 0) {
    'Archivos que requieren revisión:'
    $warningRows | Select-Object audio, speakers, speech_coverage_pct, unassigned_words, short_turn_pct, warnings | Format-Table -AutoSize
} else {
    'QA_OK: no se detectaron anomalías automáticas en los archivos terminados.'
}

if ($noteRows.Count -gt 0) {
    'Observaciones no críticas:'
    $noteRows | Select-Object audio, speakers, speech_coverage_pct, notes | Format-Table -AutoSize
}

if ($latestErrors.Count -gt 0) {
    'Errores actuales:'
    $latestErrors | Select-Object audio, detail | Format-Table -AutoSize
}
