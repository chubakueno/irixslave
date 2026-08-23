from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "README.md",
    "pipeline_transcripcion_diarizada.py",
    "ejecutar.ps1",
    "instalar.ps1",
    "configurar_huggingface.ps1",
    "requirements.txt",
    "scripts_originales/transcribir.py",
    "scripts_originales/diarizar.py",
    "scripts_originales/ejecutar_transcripcion.ps1",
    "scripts_originales/ejecutar_diarizacion.ps1",
    "scripts_originales/estado_transcripcion.ps1",
    "scripts_originales/estado_diarizacion.ps1",
    "scripts_originales/auditar_calidad.py",
    "scripts_originales/auditar_diarizacion.ps1",
    "scripts_originales/generar_transcripcion_nombrada.py",
)
SECRET_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")
FORBIDDEN_PARTS = {".venv", "modelos", "audios", "transcripciones", "diarizaciones"}
FORBIDDEN_SUFFIXES = {".bin", ".safetensors", ".ckpt", ".pt", ".pth", ".mp3", ".wav"}


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"Falta: {relative}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in FORBIDDEN_PARTS:
            errors.append(f"Directorio excluido presente: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Archivo pesado/audio excluido presente: {relative}")
        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Python inválido {relative}: {exc.msg}")
        if path.suffix.lower() in {".py", ".ps1", ".md", ".txt"}:
            content = path.read_text(encoding="utf-8-sig")
            if SECRET_PATTERN.search(content):
                errors.append(f"Posible token de Hugging Face en: {relative}")

    if errors:
        print("PAQUETE_RECHAZADO")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PAQUETE_APROBADO")
    print(f"Archivos requeridos: {len(REQUIRED)}")
    print("Modelos, audios, resultados y credenciales: ausentes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
