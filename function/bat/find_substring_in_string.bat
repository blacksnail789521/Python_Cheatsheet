@echo off
@setlocal enableextensions enabledelayedexpansion

goto :main

:echo_result
echo target: !target!
echo string: !string!
if not "!string:%target%=!"=="!string!" (
    echo We found it.
) ^
else (
    echo We couldn't find it.
)
echo -----------------
goto :eof

:main
set target=456

:: First test case (We should find it.)
set string=123456789
call :echo_result

:: Second test case (We shouldn't find it.)
set string=12345_789
call :echo_result

pause

endlocal