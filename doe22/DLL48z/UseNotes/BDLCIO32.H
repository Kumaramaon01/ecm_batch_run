///////////////////////////////////////////////////////////////////////
//
//  (c) Copyright JJ Hirsch and Associates
//      Refer to root file CopyrightNotification.txt for details.
//
///////////////////////////////////////////////////////////////////////




#ifndef BDLCIO32_H
#define BDLCIO32_H

/* #include "ExpCode/BDLCalls.h" */  /* SAC 10/24/01 */


#ifdef BDL_MS
#define  BDLCallConvPRC    __declspec(dllexport) __cdecl
#define  BDLCallConvExp    __declspec(dllexport) __stdcall
#else
#define  BDLCallConvPRC    __declspec(dllexport _cdecl)
#define  BDLCallConvExp    __declspec(dllexport _stdcall)
#endif


#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */


long BDLCallConvExp BDLCIO32_Init();
long BDLCallConvPRC BDLCIO32_CDInit();
long       _stdcall BDLCIO32_VBInit();

/* SAC 11/20/01 - Added the following exports to enable multiple versions of simulation DLL to be utilized by a single installation */
long BDLCallConvExp BDLCIO32_InitByName(   const char* pszBDLDLLName );
long BDLCallConvPRC BDLCIO32_CDInitByName( const char* pszBDLDLLName );
long       _stdcall BDLCIO32_VBInitByName( const char* pszBDLDLLName );

long BDLCallConvExp BDLCIO32_Exit();
long BDLCallConvPRC BDLCIO32_CDExit();
long       _stdcall BDLCIO32_VBExit();

long BDLCallConvExp BDLCIO32_GetDOE2Version();    /* SAC 10/25/04 */
long BDLCallConvPRC BDLCIO32_CDGetDOE2Version();
long       _stdcall BDLCIO32_VBGetDOE2Version();

long BDLCallConvExp BDLCIO32_ReadInput(   const char* workDir, const char* fileDir,
                                          const char* fileName, const char* libFileName,
                                          long lNoScrnMsg, int bWriteNHKFile,
                                          void* pCallbackFunc );
long BDLCallConvPRC BDLCIO32_CDReadInput( const char* workDir, const char* fileDir,
                                          const char* fileName, const char* libFileName,
                                          long lNoScrnMsg, int bWriteNHKFile,
                                          void* pCallbackFunc );
long       _stdcall BDLCIO32_VBReadInput( const char* workDir, const char* fileDir,
                                          const char* fileName, const char* libFileName,
                                          long lNoScrnMsg, int bWriteNHKFile,
                                          void* pCallbackFunc );


#ifdef __cplusplus
}
#endif


#endif // BDLCIO32_H
