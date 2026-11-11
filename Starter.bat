@echo off@echo off
chcp 65001 > nul
title MY_AI V2 - Transformer Control Panel

:menu
cls
echo ========================================================
echo    MY_AI V2 (PyTorch + DirectML / AMD RX 580)
echo ========================================================
echo.
echo   [1] Запустить веб-интерфейс (Streamlit: app.py)
echo   [2] Запустить обучение из консоли (train_v2.py)
echo   [3] Запустить генерацию из консоли (generate_v2.py)
echo   [4] Выход
echo.
set /p choice="Выбери действие (1-4): "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto train
if "%choice%"=="3" goto generate
if "%choice%"=="4" exit

goto menu

:streamlit
echo.
echo Запуск Streamlit интерфейса...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
goto menu

:train
echo.
echo Запуск консольного обучения (train_v2.py)...
".venv\Scripts\python.exe" train_v2.py
pause
goto menu

:generate
echo.
echo Запуск консольной генерации (generate_v2.py)...
".venv\Scripts\python.exe" generate_v2.py
pause
goto menu
