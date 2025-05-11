@echo off
REM Crear entorno virtual
python -m venv venv

REM Activar entorno virtual
call venv\Scripts\activate

REM Instalar dependencias
pip install requests beautifulsoup4 ollama

REM Guardar lista de dependencias
pip freeze > requirements.txt

echo Entorno virtual configurado y dependencias instaladas.
pause
