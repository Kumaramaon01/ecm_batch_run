echo OFF
if exist c:\doe22\exe%1 goto :test2
echo *ERROR: Version directory c:\doe22\%1does not exist.
echo *       Version that exist include:
dir /w c:\doe22\exe*
goto :nogood
:test2
set doedir=c:\doe22\exe%1
:shift
shift
goto :checkinp
:nogood
echo *ERROR: First parameter MUST specify an available version to run 
:nogood1
echo *
echo *      Correct useage of command as follows:
echo *       
echo *run22 version input weather 
echo *       
echo *      where   version is the DOE-2.2 version you want to run as follows:
echo *                    verno      - a standard release
echo *                    verno.ref  - a refrigeration release
echo *                    these must be located in the \doe22 directory
echo *       
echo *              input is the name of the BDL input file WITHOUT the required
echo *                    .inp extension
echo *       
echo *              weather is the packed DOE-2 weather file WITHOUT the 
echo *                    required .bin extension; these files must be located
echo *                    in the \doe22\weather directory
echo *       
goto :end
:checkinp
if exist %1.inp goto :checkwth
echo *ERROR: Input file (%1.inp) does not exist.
echo *       Input files to use include (leave off .inp extension):
dir /w *.inp
goto :nogood1
:checkwth
if exist c:\doe22\weather\%2.bin goto :start
echo *ERROR: Weather file (c:\doe22\weather\%2) not in library.
echo *       Weather files to use include (leave off .bin extension):
dir /w c:\doe22\weather\packed\*.bin
goto :nogood1
:start
echo ===== doe2.2: input file : %1 ,  weather : %2
echo ===== DOE-2.2 version: %doedir% =====
mkdir  %1.tmp >nul
cd     %1.tmp
copy  ..\%1.inp         INPUT2.TMP >nul
COPY  %doedir%\BDLKEY.BIN     BDLKEY.BIN >nul
copy  %doedir%\BDLLIB.dat     BDLLIB.dat >nul
copy  %doedir%\BDLDFT.dat     BDLDFT.dat >nul
%doedir%\doebdl
del   BDLKEY.BIN >nul
del   BDLLIB.dat >nul
del   BDLDFT.dat >nul
del   INPUT2.TMP >nul
copy     DOEBDL.OUT ..\%1.BDL >nul
if errorlevel 1 goto :error
COPY  c:\doe22\weather\%2.bin  WEATHER.BIN >nul
COPY  %doedir%\HDRFIL.BIN  HDRFIL.BIN >nul
COPY  %doedir%\TDVCTZ.BIN  TDVCTZ.BIN >nul
%doedir%\doesim
copy     DOESIM.OUT ..\%1.SIM >nul
:error
copy     DOEBDL.LOG + DOESIM.LOG ..\%1.log >nul
del *.dat >nul
del *.bin >nul
del *.tmp >nul
del *.log >nul
del *.out >nul
del for??? >nul
cd      ..
rmdir   %1.tmp >nul
echo ===== doe22 %1.inp %2.bin completed =====
:end
echo ON
