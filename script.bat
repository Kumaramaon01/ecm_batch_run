@ECHO off
setlocal enabledelayedexpansion
:: Script to run DOE2.1E simulations for specific .inp files based on weather
:: Only runs .INP files that contain the extracted weather location in their name

set inp_dir=%1
set weather= %2
set doe_cmd=c:\doe22\doe22.bat exe48z

call cd %inp_dir%

set "file_count=0"
:: Run simulations only for .INP files that contain the extracted weather name
for /r %inp_dir% %%f in (*.inp) do (
  set /A file_count+=1 >nul
  echo Running file !file_count!: %%~nf.inp >nul
  :: Run the DOE2 command
  call %doe_cmd% %%~pf%%~nf %weather% >nul
)

echo All simulations completed.
endlocal
:: pause
