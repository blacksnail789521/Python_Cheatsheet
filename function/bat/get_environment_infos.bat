@echo off
@setlocal enableextensions enabledelayedexpansion

echo bat_file_name:        %~n0
echo computer_name:        !COMPUTERNAME!
echo date:                 !DATE!
echo date (first 10 char): !DATE:~0,10!
echo time:                 !TIME!
echo time (first 10 8):    !TIME:~0,8!

set bat_file_name=%~n0
title !COMPUTERNAME!_!DATE:~0,10!_!TIME:~0,8!_!bat_file_name!

pause

endlocal