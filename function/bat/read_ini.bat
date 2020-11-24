@echo off
@setlocal enableextensions enabledelayedexpansion

:: Get ini_path
for %%i in ("%~dp0") do set exec_path=%%~fi
set ini_path="!exec_path!\config.ini"

:: Find target's value in ini.
for /f "usebackq delims=" %%a in (!ini_path!) do (
    set line=%%a
	if "!line:~0,1!"=="[" (
	    set section_line=!line!
	) ^
	else (
	    for /f "tokens=1,2 delims==" %%b in (!line!) do (
		    if not "!section_line:%target%=!"=="!section_line!" (
			    :: We found target in section_line.
				set current_key=%%b
				set current_value=%%c
				if !current_key!==name (
				    set name=!current_value!
					echo name: !name!
				)
				if !current_key!==age (
				    set age=!current_value!
					echo age:  !age!
				)
			)
		)
	)
)

pause

endlocal