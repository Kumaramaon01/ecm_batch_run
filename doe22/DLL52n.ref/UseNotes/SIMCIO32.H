#ifndef SIMCIO32X_H
#define SIMCIO32X_H


/**********************************************************************/
/*                                                                    */
/*   (c) Copyright 1999, Scott A. Criswell.  All Rights Reserved.     */
/*                                                                    */
/*   This computer program is protected from copying and infringement */
/*   by Federal Law under U.S.C. Title 17.                            */
/*                                                                    */
/*   Unpublished rights reserved under the copyright laws of the      */
/*   United States.                                                   */
/*                                                                    */
/**********************************************************************/


#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */


long __declspec(dllexport) __cdecl SimCIO32_Init();
long                      _stdcall SimCIO32_VBInit();

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
