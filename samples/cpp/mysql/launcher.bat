@echo off
rem ============================================================================
rem Copyright 2017-2019 Intel Corporation.
rem
rem This software  and the related  documents  are Intel  copyrighted materials,
rem and your use  of them is  governed by the express  license under  which they
rem were provided to you (License).  Unless the License provides otherwise,  you
rem may not use,  modify,  copy, publish,  distribute, disclose or transmit this
rem software or the related documents without Intel's prior written permission.
rem
rem This software and the related documents are provided as is,  with no express
rem or implied warranties,  other  than those that  are expressly  stated in the
rem License.
rem
rem License:
rem http://software.intel.com/en-us/articles/intel-sample-source-code-license-a
rem greement/
rem ============================================================================

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

if "%ARCH%"=="intel64" set ODBC_PATH="C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\x64"
if "%ARCH%"=="ia32"    set ODBC_PATH="C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\x86"

if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-nologo -w
set LFLAGS=-nologo
set LIB_DAAL=daal_core.lib daal_thread.lib
set LIB_DAAL_DLL=daal_core_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb.lib tbbmalloc.lib impi.lib
set LFLAGS_DAAL_DLL=daal_core_dll.lib
set ODBC_LIB=%ODBC_PATH%/Odbc32.lib
echo %ODBC_LIB%
set MYSQL_LOGFILE=.\%RESULT_DIR%\build_mysql.log
if not "%RMODE%"=="run" (
    if exist %MYSQL_LOGFILE% del /Q /F %MYSQL_LOGFILE%
)
set MYSQL_CPP_PATH=sources
if not defined MYSQL_SAMPLE_LIST (
    call .\daal.lst.bat
)

setlocal enabledelayedexpansion enableextensions

for %%T in (%MYSQL_SAMPLE_LIST%) do (
    if not "%RMODE%"=="run" (
        echo call icl -c %CFLAGS% %MYSQL_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %MYSQL_LOGFILE%
        call      icl -c %CFLAGS% %MYSQL_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %MYSQL_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %ODBC_LIB% -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %MYSQL_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %ODBC_LIB% -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %MYSQL_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %ODBC_LIB% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %MYSQL_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %ODBC_LIB% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %MYSQL_LOGFILE%
    )
    if not "%RMODE%"=="build" (
        for %%U in (%%T %%T_dll) do (
            if exist .\%RESULT_DIR%\%%U.exe (
                .\%RESULT_DIR%\%%U.exe "%DAAL_MYSQL_SAMPLE_CONNECTION_STRING%" 1>.\%RESULT_DIR%\%%U.res 2>&1
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
