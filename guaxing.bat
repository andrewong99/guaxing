@echo off
setlocal
title guaxing  -  Gua Figure Geometry Builder
cd /d "%~dp0"

set "SCRIPT=%~dp0guaxing.py"
if not exist "%SCRIPT%" goto noscript

rem ------------------------------------------------------------------
rem  Locate a working Python 3.  Prefer the "py" launcher, then python.
rem  Each candidate is probed, not just found on PATH, so a broken
rem  launcher or a Microsoft Store stub does not get picked.
rem ------------------------------------------------------------------
set "PY="
where py >nul 2>nul
if errorlevel 1 goto trypython
py -3 -c "import sys" >nul 2>nul
if errorlevel 1 goto trypython
set "PY=py -3"
goto gotpy

:trypython
where python >nul 2>nul
if errorlevel 1 goto nopython
python -c "import sys" >nul 2>nul
if errorlevel 1 goto nopython
set "PY=python"

:gotpy
rem ------------------------------------------------------------------
rem  guaxing.py needs only PyQt6 for the GUI.  The geometry kernel,
rem  the SVG export and --selftest are pure standard library.
rem ------------------------------------------------------------------
%PY% -c "import PyQt6" >nul 2>nul
if not errorlevel 1 goto run

echo.
echo   PyQt6 is missing - installing it now. This happens only once.
echo.
%PY% -m pip install PyQt6
%PY% -c "import PyQt6" >nul 2>nul
if not errorlevel 1 goto run
echo.
echo   Retrying with --user ...
echo.
%PY% -m pip install --user PyQt6
%PY% -c "import PyQt6" >nul 2>nul
if not errorlevel 1 goto run
goto noqt

:run
rem ------------------------------------------------------------------
rem  This console window stays open behind the GUI so that a Python
rem  traceback is visible if something goes wrong.  To launch with no
rem  console at all, replace the next line with:
rem      start "" pythonw "%SCRIPT%"
rem ------------------------------------------------------------------
%PY% "%SCRIPT%"
if errorlevel 1 goto crashed
goto done

:crashed
echo.
echo   guaxing.py exited with an error - the traceback is above.
echo.
pause
goto done

:noscript
echo.
echo   guaxing.py was not found in this folder:
echo     %~dp0
echo   Keep guaxing.bat next to guaxing.py.
echo.
pause
goto done

:nopython
echo.
echo   No working Python 3 was found.
echo   Install it from  https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" during setup.
echo.
pause
goto done

:noqt
echo.
echo   PyQt6 could not be installed automatically. Run this by hand:
echo      %PY% -m pip install PyQt6
echo.
pause
goto done

:done
endlocal
