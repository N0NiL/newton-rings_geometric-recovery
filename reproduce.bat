@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
rem One-click quick reproduction (Windows). Optional: reproduce.bat --with-manet
cd /d "%~dp0"
set "PYTHONUTF8=1"

set "WITH_MANET=0"
for %%A in (%*) do (
  if /I "%%~A"=="--with-manet" set "WITH_MANET=1"
)

echo === Newton-rings geometry: quick reproduce ===
echo Root: %CD%
echo.

rem ---- 1) Python 3.11 ----
set "PY="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY (
  where python >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%V in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do set "VER=%%V"
    if "!VER!"=="3.11" set "PY=python"
  )
)
if not defined PY (
  echo [FAIL] Python 3.11 not found. Install 3.11 and ensure `py -3.11` or `python` is on PATH.
  exit /b 1
)
echo [OK] using: %PY%
%PY% -c "import sys; print('     ', sys.version)"
echo.

rem ---- 2) venv + requirements ----
if not exist ".venv\Scripts\python.exe" (
  echo [..] creating .venv
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [FAIL] could not create .venv
    exit /b 1
  )
) else (
  echo [OK] .venv already exists
)
set "VPY=%CD%\.venv\Scripts\python.exe"
echo [..] pip install -r requirements.txt
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [FAIL] pip upgrade failed
  exit /b 1
)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [FAIL] pip install -r requirements.txt failed
  exit /b 1
)
echo [OK] dependencies installed
echo.

set "LOGDIR=%CD%\results\reproduce_logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "FAILED="

rem ---- 3) four quick checks ----
call :run_step 1 "scripts/reproduce_tables.py" "%LOGDIR%\01_reproduce_tables.log"
if errorlevel 1 set "FAILED=!FAILED! [1]reproduce_tables"
if exist "%LOGDIR%\01_reproduce_tables.log" (
  findstr /C:"Table 1 match." "%LOGDIR%\01_reproduce_tables.log" >nul
  if errorlevel 1 (
    echo [FAIL] step 1: did not print "Table 1 match."
    set "FAILED=!FAILED! [1]Table1_match"
  ) else (
    echo [OK] Table 1 match. found in step 1 log
  )
) else (
  echo [FAIL] step 1: log missing
  set "FAILED=!FAILED! [1]Table1_match"
)

call :run_step 2 "code/eval/stats.py" "%LOGDIR%\02_stats.log"
if errorlevel 1 set "FAILED=!FAILED! [2]stats"

call :run_step 3 "code/bench/run_score_sensitivity.py" "%LOGDIR%\03_score_sensitivity.log"
if errorlevel 1 set "FAILED=!FAILED! [3]score_sensitivity"

call :run_step 4 "code/eval/eval_c1_c4_reviews.py --limit 1" "%LOGDIR%\04_c1_c4_limit1.log"
if errorlevel 1 set "FAILED=!FAILED! [4]eval_c1_c4"

rem ---- 4) optional MA-Net smoke ----
if "!WITH_MANET!"=="1" (
  echo.
  echo === --with-manet: one-frame MA-Net smoke ===
  if not exist "weights\50distill_last.pt" (
    echo [FAIL] missing weights\50distill_last.pt
    set "FAILED=!FAILED! [manet]missing_weights"
  ) else (
    call :run_step manet "code/bench/run_comparison_now.py --limit 1" "%LOGDIR%\manet_comparison_limit1.log"
    if errorlevel 1 set "FAILED=!FAILED! [manet]comparison"
  )
) else (
  echo.
  echo [skip] MA-Net smoke ^(pass --with-manet to enable^)
)

echo.
if defined FAILED (
  echo ========================================
  echo FAILED steps:!FAILED!
  echo Logs: %LOGDIR%
  echo ========================================
  exit /b 1
)

echo ========================================
echo 全部通过
echo All quick checks passed.
echo Logs: %LOGDIR%
echo ========================================
exit /b 0

:run_step
set "STEP=%~1"
set "CMD=%~2"
set "LOG=%~3"
echo [..] step %STEP%: python %CMD%
"%VPY%" %CMD% >"%LOG%" 2>&1
set "RC=!errorlevel!"
if not "!RC!"=="0" (
  echo [FAIL] step %STEP% exited with code !RC!
  echo        see %LOG%
  exit /b 1
)
echo [OK] step %STEP%
exit /b 0
