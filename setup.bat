@echo off
echo Creando entorno virtual...
python -m venv venv

echo Activando entorno virtual...
call venv\Scripts\activate

echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo Instalación completada.
pause
