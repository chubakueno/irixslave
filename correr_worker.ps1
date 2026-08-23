param(
    [switch]$Live
)

$ErrorActionPreference = 'Stop'
$venv = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Falta .venv. Corre primero .\instalar.ps1"
}

# Evita que Windows suspenda el sistema/pantalla mientras el worker corre,
# sin tocar la configuración de energía de forma permanente (equivalente a
# 'caffeinate -s' en macOS). Se libera automáticamente al salir del script.
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class SleepBlocker {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
}
"@

[SleepBlocker]::SetThreadExecutionState(
    [SleepBlocker]::ES_CONTINUOUS -bor [SleepBlocker]::ES_SYSTEM_REQUIRED -bor [SleepBlocker]::ES_DISPLAY_REQUIRED
) | Out-Null

Write-Host "Worker corriendo (el sistema no se suspenderá mientras esta ventana esté abierta)."
Write-Host "Presiona Ctrl+C para detener."

try {
    if ($Live) {
        & $python (Join-Path $PSScriptRoot 'worker_transcripcion.py') --live
    } else {
        & $python (Join-Path $PSScriptRoot 'worker_transcripcion.py')
    }
} finally {
    [SleepBlocker]::SetThreadExecutionState([SleepBlocker]::ES_CONTINUOUS) | Out-Null
}
