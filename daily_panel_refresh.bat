@echo off
rem 每日行情面板增量刷新（纯接口路径，不依赖 THS 导出）
rem 由计划任务 QuantPanelDailyRefresh 在交易日 18:40 调用；日志见 outputs\logs\panel_refresh_daily.log
setlocal
set "PY=C:\Users\86176\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python.exe"
set "PROJ=D:\workbuddy_strategy\workbuddy_strategy\high_risk_quant_model3"
if not exist "%PROJ%\outputs\logs" mkdir "%PROJ%\outputs\logs"
"%PY%" "%PROJ%\daily_panel_refresh.py" %* >> "%PROJ%\outputs\logs\panel_refresh_daily.log" 2>&1
exit /b %ERRORLEVEL%
