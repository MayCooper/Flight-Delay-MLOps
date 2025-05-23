:: ------------------------------------------------------------------------------
:: Author: May Cooper
:: Script: activate.bat
::
:: This batch script sets up a Windows command-line environment to activate
:: a Python virtual environment. It adjusts the PATH, prompt, and encoding
:: to ensure proper execution of Python scripts inside the specified venv.
:: ------------------------------------------------------------------------------

@echo off

rem Ensure UTF-8 encoding by temporarily switching the code page
for /f "tokens=2 delims=:." %%a in ('"%SystemRoot%\System32\chcp.com"') do (
    set _OLD_CODEPAGE=%%a
)
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" 65001 > nul
)

rem Set path to the virtual environment
set VIRTUAL_ENV=W:\MayCooperStation\New Documents\WGU\Masters in Data Analytics - Data Science\Deployment - D602\Task 3\d602-deployment-task-3\airline_api_env

rem Set default prompt if not already set
if not defined PROMPT set PROMPT=$P$G

rem Restore previous virtual environment prompt and Python settings if they exist
if defined _OLD_VIRTUAL_PROMPT set PROMPT=%_OLD_VIRTUAL_PROMPT%
if defined _OLD_VIRTUAL_PYTHONHOME set PYTHONHOME=%_OLD_VIRTUAL_PYTHONHOME%

rem Save current prompt and Python settings
set _OLD_VIRTUAL_PROMPT=%PROMPT%
set PROMPT=(airline_api_env) %PROMPT%

if defined PYTHONHOME set _OLD_VIRTUAL_PYTHONHOME=%PYTHONHOME%
set PYTHONHOME=

rem Update PATH to include Scripts folder of the virtual environment
if defined _OLD_VIRTUAL_PATH set PATH=%_OLD_VIRTUAL_PATH%
if not defined _OLD_VIRTUAL_PATH set _OLD_VIRTUAL_PATH=%PATH%

set PATH=%VIRTUAL_ENV%\Scripts;%PATH%
set VIRTUAL_ENV_PROMPT=(airline_api_env)

:END
rem Restore original code page after execution
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" %_OLD_CODEPAGE% > nul
    set _OLD_CODEPAGE=
)
