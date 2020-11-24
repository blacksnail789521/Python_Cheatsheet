@echo off
@setlocal enableextensions enabledelayedexpansion

goto :main

:echo_people
set name=%~1
set age=%~2
echo name: !name!
echo age:  !age!
echo -----------------
goto :eof

:main
call :echo_people Jason 26
call :echo_people Carol 27
call :echo_people Brian 32

pause

endlocal