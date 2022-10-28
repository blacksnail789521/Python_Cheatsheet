@echo off
@setlocal enableextensions enabledelayedexpansion

some_error

if !ERRORLEVEL! neq 0 (
    echo handling error
)

pause

endlocal