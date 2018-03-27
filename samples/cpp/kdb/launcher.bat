@echo off
rem==============================================================
rem
rem SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
rem http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
rem
rem Copyright 2017-2018 Intel Corporation
rem
rem THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
rem NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
rem PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
rem
rem =============================================================

::  Content:
::     Intel(R) Data Analytics Acceleration Library samples creation and run
::******************************************************************************

set ARCH=%1
set RMODE=%2

set errorcode=0

if "%1"=="help" (
    goto :Usage
)

if not "%ARCH%"=="ia32" if not "%ARCH%"=="intel64" (
    echo Bad first argument, must be ia32 or intel64
    set errorcode=1
    goto :Usage
)

if not "%RMODE%"=="build" if not "%RMODE%"=="run" if not "%RMODE%"=="" (
    echo Bad second argument, must be build or run
    set errorcode=1
    goto :Usage
)

goto :CorrectArgs

:Usage
echo Usage: launcher.bat ^{arch^|help^} [rmode]
echo arch  - can be ia32 or intel64
echo rmode - optional parameter, can be build (for building samples only) or
echo         run (for running samples only).
echo         If not specified build and run are performed.
echo help  - print this message
exit /b errorcode

:CorrectArgs

set RESULT_DIR=_results\%ARCH%

if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-nologo -w -DDAAL_CHECK_PARAMETER /I %KDB_HEADER_PATH%
set LFLAGS=-nologo
set LIB_DAAL=daal_core.lib daal_thread.lib
set LIB_DAAL_DLL=daal_core_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb.lib tbbmalloc.lib impi.lib
set LFLAGS_DAAL_DLL=daal_core_dll.lib
set KDB_LOGFILE=.\%RESULT_DIR%\build_kdb.log
if not "%RMODE%"=="run" (
    if exist %KDB_LOGFILE% del /Q /F %KDB_LOGFILE%
)
set KDB_CPP_PATH=sources
if not defined KDB_SAMPLE_LIST (
    call .\daal.lst.bat
)

setlocal enabledelayedexpansion enableextensions

for %%T in (%KDB_SAMPLE_LIST%) do (
    if not "%RMODE%"=="run" (
        echo call icl -c %CFLAGS% %KDB_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %KDB_LOGFILE%
        call      icl -c %CFLAGS% %KDB_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %KDB_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %KDB_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %KDB_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %KDB_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %KDB_LOGFILE%
    )
    if not "%RMODE%"=="build" (
        for %%U in (%%T %%T_dll) do (
            if exist .\%RESULT_DIR%\%%U.exe (
                .\%RESULT_DIR%\%%U.exe 1>.\%RESULT_DIR%\%%U.res 2>&1
                if "!errorlevel!" == "0" (
                    echo %time% PASSED %%U
                ) else (
                    echo %time% FAILED %%U with errno !errorlevel!
                )
            ) else (
                echo %time% BUILD FAILED %%U
            )
        )
    )
)

endlocal

exit /B %ERRORLEVEL%

:out
