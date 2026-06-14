@echo off
chcp 65001 > nul
echo ⛳ ゴルフ場 精密天気予報 を起動しています...
echo.
echo ブラウザが自動で開きます。開かない場合は http://localhost:8501 にアクセスしてください。
echo.
echo 終了するには このウィンドウを閉じてください。
echo.
python -m streamlit run "%~dp0app.py" --server.headless false
pause
