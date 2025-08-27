

///////////////////////////////////////////////////////////////////////
//
//  (c) Copyright JJ Hirsch and Associates
//      Refer to root file CopyrightNotification.txt for details.
//
///////////////////////////////////////////////////////////////////////


#ifndef SIMCIO32X_H
#define SIMCIO32X_H



#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */


long __declspec(dllexport) __cdecl SimCIO32_Init();
long                      _stdcall SimCIO32_VBInit();

/* SAC 11/20/01 - Added the following exports to enable multiple versions of simulation DLL to be utilized by a single installation */
long __declspec(dllexport) __cdecl SimCIO32_InitByName(   const char* pszSimDLLName );
long                      _stdcall SimCIO32_VBInitByName( const char* pszSimDLLName );

long __declspec(dllexport) __cdecl SimCIO32_Exit();
long                      _stdcall SimCIO32_VBExit();

typedef long (CALLBACK* PUICallbackFunc) ( long l1, long l2, long l3, long l4, long l5 );

long __declspec(dllexport) __cdecl SimCIO32_RunSimulation( const char* workDir, const char* fileDir, const char* fileName,
                                                           const char* wthrFileName, long lNoScrnMsg, PUICallbackFunc pCallbackFunc );
long                      _stdcall SimCIO32_VBRunSimulation( const char* workDir, const char* fileDir, const char* fileName,
                                                             const char* wthrFileName, long lNoScrnMsg, PUICallbackFunc pCallbackFunc );


#ifdef __cplusplus
}
#endif


#endif // SIMCIO32X_H
