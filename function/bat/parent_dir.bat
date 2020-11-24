@echo off
@setlocal enableextensions enabledelayedexpansion

for %%i in ("%~dp0") do set exec_path=%%~fi
echo "!exec_path!"

for %%i in ("%~dp0\..") do set exec_path=%%~fi
echo "!exec_path!"

for %%i in ("%~dp0\..\..") do set exec_path=%%~fi
echo "!exec_path!"

pause

endlocal