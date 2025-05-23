:: ------------------------------------------------------------------------------
:: Author: May Cooper
:: Script: deactivate.bat
::
:: This script restores the original environment variables and shell prompt
:: after deactivating a Python virtual environment in Windows CMD.
:: It cleans up any changes made during activation.
:: ------------------------------------------------------------------------------

@echo off

if defined _OLD_VIRTUAL_PROMPT (
    set "PROMPT=%_OLD_VIRTUAL_PROMPT%"
)
set _OLD_VIRTUAL_PROMPT=

if defined _OLD_VIRTUAL_PYTHONHOME (
    set "PYTHONHOME=%_OLD_VIRTUAL_PYTHONHOME%"
    set _OLD_VIRTUAL_PYTHONHOME=
)

if defined _OLD_VIRTUAL_PATH (
    set "PATH=%_OLD_VIRTUAL_PATH%"
)

set _OLD_VIRTUAL_PATH=

set VIRTUAL_ENV=
set VIRTUAL_ENV_PROMPT=

:END
