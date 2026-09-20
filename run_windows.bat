@echo off
setlocal
cd /d "%~dp0"
if "%HANA_HOST%"=="" set "HANA_HOST=127.0.0.1"
if "%HANA_PORT%"=="" set "HANA_PORT=7860"
if "%HANA_DTYPE%"=="" set "HANA_DTYPE=auto"
if "%HANA_REMOVE_WEIGHT_NORM%"=="" set "HANA_REMOVE_WEIGHT_NORM=1"
if "%HANA_TORCH_COMPILE%"=="" set "HANA_TORCH_COMPILE=0"
if "%HANA_QUEUE_SIZE%"=="" set "HANA_QUEUE_SIZE=4"
if "%HANA_CACHE_ITEMS%"=="" set "HANA_CACHE_ITEMS=32"
if "%HANA_CACHE_MAX_MB%"=="" set "HANA_CACHE_MAX_MB=64"
python -m uvicorn api.server:app --host %HANA_HOST% --port %HANA_PORT% --workers 1
endlocal
