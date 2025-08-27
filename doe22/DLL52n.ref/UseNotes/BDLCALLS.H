/*****  BDLCalls.H  *****/

///////////////////////////////////////////////////////////////////////
//
//  (c) Copyright JJ Hirsch and Associates
//      Refer to root file CopyrightNotification.txt for details.
//
///////////////////////////////////////////////////////////////////////



#ifndef BDLCalls_H
#define BDLCalls_H

#include <stdio.h>


#ifdef BDL_MS
#define  BDLFileOpenFlag      _SH_DENYNO
#define  BDLFileOpenWRFlag    _SH_DENYWR
#define  BDLCallConvention    __declspec(dllexport) __stdcall
#define  BDLCallConvNoExport  __stdcall
#define  BDLCallConvInline    __inline
#elif __WATCOMC__
#define  BDLFileOpenFlag      SH_DENYNO
#define  BDLFileOpenWRFlag    SH_DENYWR
#define  BDLCallConvention    __export __stdcall
#define  BDLCallConvNoExport  __stdcall
#define  BDLCallConvInline    __inline
#else
#define  BDLFileOpenFlag      SH_DENYNO
#define  BDLFileOpenWRFlag    SH_DENYWR
#define  BDLCallConvention    __declspec(dllexport _stdcall)
#define  BDLCallConvNoExport  __declspec(__stdcall)
#define  BDLCallConvInline    __inline
#endif

#endif /* BDLCalls_H */
