$ErrorActionPreference = 'Stop'

$modelUrl = 'https://huggingface.co/pyannote/speaker-diarization-community-1'
Write-Host '1. Abre esta página, inicia sesión y acepta las condiciones del modelo:'
Write-Host $modelUrl
Write-Host ''
Write-Host '2. Crea un token READ en https://huggingface.co/settings/tokens'
Write-Host '3. Pega el token en el prompt seguro que aparecerá a continuación.'
Write-Host '   No lo escribas en un chat ni lo guardes dentro de los scripts.'
Write-Host ''

& "$PSScriptRoot\.venv\Scripts\hf.exe" auth login
if ($LASTEXITCODE -ne 0) {
    throw 'No se pudo guardar la autorización de Hugging Face.'
}

Write-Host ''
Write-Host 'Autorización guardada en el almacén local de Hugging Face.'
