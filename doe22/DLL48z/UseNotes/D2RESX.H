///////////////////////////////////////////////////////////////////////
//
//  (c) Copyright JJ Hirsch and Associates
//      Refer to root file CopyrightNotification.txt for details.
//
///////////////////////////////////////////////////////////////////////

// D2ResX.h




// Valid IDs for identifying BDL components from NHK files
#define  BDLNHKCOM_COMPONENT_COST     2001
#define  BDLNHKCOM_UTILITY_RATE       2005
#define  BDLNHKCOM_BLOCK_CHARGE       2006
#define  BDLNHKCOM_RATCHET            2007
#define  BDLNHKCOM_SYSTEM             2010
#define  BDLNHKCOM_ZONE               2019
#define  BDLNHKCOM_EXTERIOR_WALL      2020
#define  BDLNHKCOM_UNDERGROUND_WALL   2040
#define  BDLNHKCOM_INTERIOR_WALL      2046
#define  BDLNHKCOM_WINDOW             2050
#define  BDLNHKCOM_DOOR               2055
#define  BDLNHKCOM_PUMP               2061
#define  BDLNHKCOM_CIRCULATION_LOOP   2062
#define  BDLNHKCOM_CHILLER            2063
#define  BDLNHKCOM_BOILER             2064
#define  BDLNHKCOM_DW_HEATER          2065
#define  BDLNHKCOM_HEAT_REJECTION     2066
#define  BDLNHKCOM_ELEC_GENERATOR     2068
#define  BDLNHKCOM_THERMAL_STORAGE    2069
#define  BDLNHKCOM_SPACE              2070
#define  BDLNHKCOM_ELEC_METER         2085
#define  BDLNHKCOM_FUEL_METER         2086
#define  BDLNHKCOM_STEAM_METER        2087
#define  BDLNHKCOM_CHW_METER          2088
#define  BDLNHKCOM_MATERIALS_COST     2091
#define  BDLNHKCOM_GROUND_LOOP_HX     2092    // SAC 4/16/10 - added to facilitate retrieval of GROUND-LOOP-HX results


// Valid IDs for identifying what DOE-2 Non-Hourly results to retrieve
#define  D2RFT_NHRLY_LOADS  0
#define  D2RFT_NHRLY_HVAC   1
#define  D2RFT_NHRLY_ECON   2
// RW 11/21/00 Note -> for now lets set it up so that the hourly filetype is determined
#define  D2RFT_HRLY_LOADS     3
#define  D2RFT_HRLY_HVAC      4
// RW 05/13/02 Note -> for now lets set it up so that the Weather filetype is determined
#define  D2RFT_HRLY_WEATHER   5


#define D2R_MIN_YEAR     1900    // Min year
#define D2R_MAX_YEAR     2050    // Max year


#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

struct ResultsFilter;

// FOR ALL OF THE FOLLOWING EXPORTED FUNCTIONS:
//    const char* pInpFileName : NULL terminated input path and filename character string
//                               w/ NO trailing extension
//                               (i.e. if your input filename is "C:\My Dir\Projects\Wall Mart\Wall Mart.inp"
//                                then pInpFileName = "C:\My Dir\Projects\Wall Mart\Wall Mart"


// Return values:  >=  0 : Successful
//                   -12 : Invalid DOE-2 path passed in
//                   -13 : Error reading "NHRList.txt"
long __declspec(dllexport) __cdecl D2R_LoadNHRList( const char* pszDOE2Dir );
long                      _stdcall D2R_VBLoadNHRList( const char* pszDOE2Dir );

// Return Values:  >= 0 : Successful
//                   -1 : Invalid pInpFileName (.NHK file does not exist)
//                   -2 : Error reading <pInpFileName>.NHK file
long __declspec(dllexport) __cdecl D2R_LoadNHKData( const char* pszInpFileName );
long                      _stdcall D2R_VBLoadNHKData( const char* pszInpFileName );


// Return Values:  >  0 : Successful
//                 <= 0 : not found
long __declspec(dllexport) __cdecl D2R_BDLCommandToNHKID( int i1BDLComIdx );
long                      _stdcall D2R_VBBDLCommandToNHKID( int i1BDLComIdx );

// Return Values:  >  0 : Successful
//                 <= 0 : not found
long __declspec(dllexport) __cdecl D2R_NHKIDToBDLCommand( int iNHKID );
long                      _stdcall D2R_VBNHKIDToBDLCommand( int iNHKID );


// SAC 6/29/99
BOOL __declspec(dllexport) __cdecl D2R_GetNumItemsAndKeyIDs(   int iNHRLID, int& iNumItems, int& iReportKey, int& iRowKey );
BOOL                      _stdcall D2R_VBGetNumItemsAndKeyIDs( int iNHRLID, int& iNumItems, int& iReportKey, int& iRowKey );

// SAC 6/13/04
BOOL __declspec(dllexport) __cdecl D2R_GetNumItemsLengthTypeAndKeyIDs(   int iNHRLID, int& iNumItems, int& iLengthSum, int& iItemType, int& iReportKey, int& iRowKey );
BOOL                      _stdcall D2R_VBGetNumItemsLengthTypeAndKeyIDs( int iNHRLID, int& iNumItems, int& iLengthSum, int& iItemType, int& iReportKey, int& iRowKey );


// Arguments:
//       int iNHKCompType : value defined above as: BDLNHKCOM_***
//       int iBDLCompType : -1 to retrieve ALL components of this type,
//                          else, for a subset of "TYPE" BDL command components, specify the numeric
//                                value associated with desired TYPE symbol
//
// Return Values:  >= 0 : Successful -> # components found
//                   -1 : Invalid iNHKCompType value
long __declspec(dllexport) __cdecl D2R_GetNumComponents( int iNHKCompType, int iBDLCompType );
long                      _stdcall D2R_VBGetNumComponents( int iNHKCompType, int iBDLCompType );


// Arguments:
//       int iNHKCompType  : (see description above)
//       int iBDLCompType  : (see description above)
//       int iCompIdx      : 0-based index of the desired component
//       char* pszCompName : Pointer to the character string to be populated w/ component name.
//                           The string will be returned with a NULL terminator.
//       int iMaxNameChars : Maximum number of characters to insert into the string.
//                           (note: the current BDL max character limit is 32, but in order
//                                 to return a name of 32 characters, the input character 
//                                 string length must be >= 33 characters long to allow
//                                 space for the terminating NULL character.
//
// Return Values:  > 0 : Length of component name character string (excluding NULL terminator)
//                  -1 : Invalid iNHKCompType value
//                  -2 : Component not found
long __declspec(dllexport) __cdecl D2R_GetComponentName( int iNHKCompType, int iBDLCompType, int iCompIdx,
                                                         char* pszCompName, int iMaxNameChars );
long                      _stdcall D2R_VBGetComponentName( int iNHKCompType, int iBDLCompType, int iCompIdx,
                                                           char* pszCompName, int iMaxNameChars );

// SAC 12/11/05 - Added new copies of the above routines to gain access to TYPE of component whose name is also returned
long __declspec(dllexport) __cdecl D2R_GetComponentNameAndType( int iNHKCompType, int iBDLCompType, int iCompIdx,
                                                                char* pszCompName, int iMaxNameChars, int& iCompType );
long                      _stdcall D2R_VBGetComponentNameAndType( int iNHKCompType, int iBDLCompType, int iCompIdx,
                                                                  char* pszCompName, int iMaxNameChars, int& iCompType );



// Arguments:
//       int     iEntryID         : value defined in the first column of DOE-2\NHRList.txt
//       float*  pfData           : pointer to an array of floats or character strings to be set with result data
//       int     iMaxReturnValues : number of floats (or sets of 4 characters) defined in above array (pfData)
//       const char* pszReportKey : typically a BDL component name identifying which of multiple
//                                     reports to retrieve data from.
//       const char* pszRowKey    : a row key string used to retrieve a particular row of data
//                                     from a report.  This is typically a month abbreviation such
//                                     as "JAN" but can also be a BDL component name.
//
// Return Values:  >= 0 : Successful -> # values retrieved
//                   -1 : Invalid pInpFileName
//                   -2 : Invalid iFileType
//                   -3 : Results file does not exist
//                   -4 : Invalid pfData
//                   -5 : Invalid iEntryID
//                   -6 : iMaxReturnValues too small
//                   -7 : pszReportKey required but not provided
//                   -8 : pszRowKey required but not provided
//                   -9 : Error retrieving data from file
//                   -12 : Invalid DOE-2 directory
//                   -13 : Bad or missing DOE-2\NHRList.txt
long __declspec(dllexport) __cdecl D2R_GetSingleResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName,
                                                        int iEntryID, float* pfData,
                                                        int iMaxReturnValues, const char* pszReportKey,
                                                        const char* pszRowKey );
long                      _stdcall D2R_VBGetSingleResult( const char* pszDOE2Dir,
                                                          const char* pszInpFileName,
                                                          int iEntryID, float* pfData,
                                                          int iMaxReturnValues, const char* pszReportKey,
                                                          const char* pszRowKey );


const int  MRT_KEY_LEN = 40;

struct MultResultsType
{
   int  iEntryID;      // value defined in the first column of DOE-2\NHRList.txt
   int  iReturnValue;  // success/failure code
   char pszReportKey[ MRT_KEY_LEN ];
   char pszRowKey[    MRT_KEY_LEN ];
};

//  int iNumMRTs  MAX VALUE = 12
//  each MultResultsType structure must be retrieving the same number of values
//
// Return Values:  >= 0 - -9 : Same as above
//                   -10 : Error retrieving data from file
//                   -11 : Not all data MRTs return same number of values
//                   -12 : Invalid DOE-2 directory
//                   -13 : Bad or missing DOE-2\NHRList.txt
//                   -14 : Cannot retrieve character strings via this routine
long __declspec(dllexport) __cdecl D2R_GetMultipleResult( const char* pszDOE2Dir,
                                                          const char* pszInpFileName, int iFileType,
                                                          float* pfData, int iMaxReturnValues,
                                                          int iNumMRTs, MultResultsType* pMRTs );
long                      _stdcall D2R_VBGetMultipleResult( const char* pszDOE2Dir,
                                                            const char* pszInpFileName, int iFileType,
                                                            float* pfData, int iMaxReturnValues,
                                                            int iNumMRTs, MultResultsType* pMRTs );

// struct used for retrieving multiple results
// RW_MOD 02/02/01 Also now passed in for the KeyOnItem information
struct MultHourlyResultsType
{
   int     iPrimaryID;      // value defined in the TYPE column of DOE-2\HourlySeries.txt
   int     iSecondaryID;    // value defined in the INDEX column of DOE-2\HourlySeries.txt
   int     iReturnValue;       // success/failure code
   char    pszID[ MRT_KEY_LEN ];
};

// RW 11/21/00 Added routines for accessing the hourly results data
// Arguments:
//       int     iFileType        : value indicating the type of hourly results file being read
//       int     iPrimaryID       : value defined in the TYPE column of DOE-2\HourlySeries.txt
//       int     iSecondaryID     : value defined in the INDEX column of DOE-2\HourlySeries.txt
//       float*  pfData           : pointer to an array of floats to be set with result data
//       int     iMaxReturnValues : number of floats defined in above array (pfData)
//       const char* pszID        : typically a BDL component name to check for which data to retrieve
//       MultHourlyResultsType* pKeyItemInfo : a structure containing all of the necessary information for Key on Item capability
//                                         this set to NULL if there is no Key on Item desired.
//
// Return Values:  >= 0 : Successful -> # values retrieved
//                   -1 : Invalid pInpFileName
//                   -2 : Invalid iFileType
//                   -3 : Results file does not exist
//                   -4 : Invalid pfData
//                   -5 : Invalid iPrimaryID
//                   -5 : Invalid iSecondaryID
//                   -6 : iMaxReturnValues too small
//                   -7 : Error retrieving data from file
//                   -8 : Invalid DOE-2 directory
long __declspec(dllexport) __cdecl D2R_GetSingleHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName,
                                                        int iFileType, int iPrimaryID,
                                                        int iSecondaryID, float* pfData,
                                                        int iMaxReturnValues, const char* pszID,
                                                        ResultsFilter* pFilter, MultHourlyResultsType* pKeyItemInfo );
long                      _stdcall D2R_VBGetSingleHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName,
                                                        int iFileType, int iPrimaryID,
                                                        int iSecondaryID, float* pfData,
                                                        int iMaxReturnValues, const char* pszID, MultHourlyResultsType* pKeyItemInfo );

// Arguments:
//       int     iFileType        : value indicating the type of hourly results file being read
//       int     iPrimaryID       : value defined in the TYPE column of DOE-2\HourlySeries.txt
//       int     iSecondaryID     : value defined in the INDEX column of DOE-2\HourlySeries.txt
//       float*  pfData           : pointer to an array of floats to be set with result data
//       int     iMaxReturnValues : number of floats defined in above array (pfData)
//       const char* pszID        : typically a BDL component name to check for which data to retrieve
//       const char* pszFilterPathName       : The file/path name of the filter file that is to be read in and define the filter 
//                                             that the data is to be retrieved in.
//       MultHourlyResultsType* pKeyItemInfo : a structure containing all of the necessary information for Key on Item capability
//                                             this set to NULL if there is no Key on Item desired.
//
// Return Values:  >= 0 : Successful -> # values retrieved
//                   -1 : Invalid pInpFileName
//                   -2 : Invalid iFileType
//                   -3 : Results file does not exist
//                   -4 : Invalid pfData
//                   -5 : Invalid iPrimaryID
//                   -5 : Invalid iSecondaryID
//                   -6 : iMaxReturnValues too small
//                   -7 : Error retrieving data from file
//                   -8 : Invalid DOE-2 directory
long __declspec(dllexport) __cdecl D2R_GetSingleFilteredHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName,
                                                        int iFileType, int iPrimaryID,
                                                        int iSecondaryID, float* pfData,
                                                        int iMaxReturnValues, const char* pszID,
                                                        const char* pszFilterPathName, MultHourlyResultsType* pKeyItemInfo );
long                      _stdcall D2R_VBGetSingleFilteredHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName,
                                                        int iFileType, int iPrimaryID,
                                                        int iSecondaryID, float* pfData,
                                                        int iMaxReturnValues, const char* pszID,
                                                        const char* pszFilterPathName, MultHourlyResultsType* pKeyItemInfo );


long __declspec(dllexport) __cdecl D2R_GetMultipleHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName, int iFileType,
                                                        float* pfData, int iMaxReturnValues,
                                                        int iNumMHRTs, MultHourlyResultsType* pMHRTs,
                                                        ResultsFilter* pFilter, MultHourlyResultsType* pKeyItemInfo );

long                      _stdcall D2R_VBGetMultipleHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName, int iFileType,
                                                        float* pfData, int iMaxReturnValues,
                                                        int iNumMHRTs, MultHourlyResultsType* pMHRTs,
                                                        ResultsFilter* pFilter, MultHourlyResultsType* pKeyItemInfo );

long __declspec(dllexport) __cdecl D2R_GetMultipleFilteredHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName, int iFileType,
                                                        float* pfData, int iMaxReturnValues,
                                                        int iNumMHRTs, MultHourlyResultsType* pMHRTs,
                                                        const char* pszFilterPathName, MultHourlyResultsType* pKeyItemInfo );

long                      _stdcall D2R_VBGetMultipleFilteredHourlyResult( const char* pszDOE2Dir,
                                                        const char* pszInpFileName, int iFileType,
                                                        float* pfData, int iMaxReturnValues,
                                                        int iNumMHRTs, MultHourlyResultsType* pMHRTs,
                                                        const char* pszFilterPathName, MultHourlyResultsType* pKeyItemInfo );


// SAC 4/24/01 - Added access to various calendar functions
//               Arguments:  iYear = 1900-2050  |  iMonth = 1-12  |  iDay = 1-(31)
//               Returns:  1=Sun, 2=Mon, ... 7=Sat
int  __declspec(dllexport) __cdecl D2R_GetDayOfWeek( int iYear, int iMonth, int iDay, BOOL bCalcLeap );

// SAC 4/23/03 - Added enable date range calcs
int  __declspec(dllexport) __cdecl D2R_MakeJulian( int year, int month, int day, BOOL calcLeap );

// SAC 3/18/05 - Added new routine to facilitate reporting of DOE-2 version used to write result file
long __declspec(dllexport) __cdecl D2R_GetResultFileDOE2Version( const char* pPathFileName, char* pszDOE2Ver, int iMaxVerLen );


// SAC 10/9/08 - added some non-hourly weather data retrieval stuff to facilitate eQUEST QC reporting
struct D2WeatherHeaderBlock
{
   long  lHeader;
   char  cLocID[20];     // IWDID          LOCATION I.D.
   long  lYear;          // IWYR           YEAR
   float fLatitude;      // WLAT           LATITUDE
   float fLongitude;     // WLONG          LONGITUDE
   long  lTimeZoneNum;   // IWTZN          TIME ZONE NUMBER
   long  lSolarFlag;     // IWSOL          SOLAR FLAG         IWSOL = IWSZ + (IFTYP-1)*2 - 1
   long  lRec1Tail;
   float fClearness;     // CLN            CLEARNESS NO.
   float fGrndTemp;      // GROUND TEMP.   (DEG R)
   long  lRec2Tail;
   long  lWthrData[1536];
   long  lTrailer;
};

long __declspec(dllexport) __cdecl D2R_GetWeatherFileHeader( const char* pszInpFileName, D2WeatherHeaderBlock* pD2WthrHdr );


#ifdef __cplusplus
}
#endif
