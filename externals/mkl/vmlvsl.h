/* file: vmlvsl.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  VML/VSL function declarations
//--
*/


#ifndef __VMLVSL_H__
#define __VMLVSL_H__


#if defined(__cplusplus)
extern "C" {
#endif


typedef void * DAAL_VSLSSTaskPtr;


#if defined(_WIN64) || defined(__x86_64__)

void fpk_vml_sLn_EXHAynn(int , float* , float* );
void fpk_vml_sLn_EXLAynn(int , float* , float* );
void fpk_vml_sLn_EXEPnnn(int , float* , float* );
void fpk_vml_dLn_EXHAynn(int , double* , double* );
void fpk_vml_dLn_EXLAynn(int , double* , double* );
void fpk_vml_dLn_EXEPnnn(int , double* , double* );

void fpk_vml_sLn_U8HAynn(int , float* , float* );
void fpk_vml_sLn_U8LAynn(int , float* , float* );
void fpk_vml_sLn_U8EPnnn(int , float* , float* );
void fpk_vml_dLn_U8HAynn(int , double* , double* );
void fpk_vml_dLn_U8LAynn(int , double* , double* );
void fpk_vml_dLn_U8EPnnn(int , double* , double* );

void fpk_vml_sLn_H8HAynn(int , float* , float* );
void fpk_vml_sLn_H8LAynn(int , float* , float* );
void fpk_vml_sLn_H8EPnnn(int , float* , float* );
void fpk_vml_dLn_H8HAynn(int , double* , double* );
void fpk_vml_dLn_H8LAynn(int , double* , double* );
void fpk_vml_dLn_H8EPnnn(int , double* , double* );

void fpk_vml_sLn_E9HAynn(int , float* , float* );
void fpk_vml_sLn_E9LAynn(int , float* , float* );
void fpk_vml_sLn_E9EPnnn(int , float* , float* );
void fpk_vml_dLn_E9HAynn(int , double* , double* );
void fpk_vml_dLn_E9LAynn(int , double* , double* );
void fpk_vml_dLn_E9EPnnn(int , double* , double* );

void fpk_vml_sLn_L9HAynn(int , float* , float* );
void fpk_vml_sLn_L9LAynn(int , float* , float* );
void fpk_vml_sLn_L9EPnnn(int , float* , float* );
void fpk_vml_dLn_L9HAynn(int , double* , double* );
void fpk_vml_dLn_L9LAynn(int , double* , double* );
void fpk_vml_dLn_L9EPnnn(int , double* , double* );

void fpk_vml_sLn_B3HAynn(int , float* , float* );
void fpk_vml_sLn_B3LAynn(int , float* , float* );
void fpk_vml_sLn_B3EPnnn(int , float* , float* );
void fpk_vml_dLn_B3HAynn(int , double* , double* );
void fpk_vml_dLn_B3LAynn(int , double* , double* );
void fpk_vml_dLn_B3EPnnn(int , double* , double* );

void fpk_vml_sLn_Z0HAynn(int , float* , float* );
void fpk_vml_sLn_Z0LAynn(int , float* , float* );
void fpk_vml_sLn_Z0EPnnn(int , float* , float* );
void fpk_vml_dLn_Z0HAynn(int , double* , double* );
void fpk_vml_dLn_Z0LAynn(int , double* , double* );
void fpk_vml_dLn_Z0EPnnn(int , double* , double* );


void fpk_vml_sExp_EXHAynn(int , float* , float* );
void fpk_vml_sExp_EXLAynn(int , float* , float* );
void fpk_vml_sExp_EXEPnnn(int , float* , float* );
void fpk_vml_dExp_EXHAynn(int , double* , double* );
void fpk_vml_dExp_EXLAynn(int , double* , double* );
void fpk_vml_dExp_EXEPnnn(int , double* , double* );

void fpk_vml_sExp_U8HAynn(int , float* , float* );
void fpk_vml_sExp_U8LAynn(int , float* , float* );
void fpk_vml_sExp_U8EPnnn(int , float* , float* );
void fpk_vml_dExp_U8HAynn(int , double* , double* );
void fpk_vml_dExp_U8LAynn(int , double* , double* );
void fpk_vml_dExp_U8EPnnn(int , double* , double* );

void fpk_vml_sExp_H8HAynn(int , float* , float* );
void fpk_vml_sExp_H8LAynn(int , float* , float* );
void fpk_vml_sExp_H8EPnnn(int , float* , float* );
void fpk_vml_dExp_H8HAynn(int , double* , double* );
void fpk_vml_dExp_H8LAynn(int , double* , double* );
void fpk_vml_dExp_H8EPnnn(int , double* , double* );

void fpk_vml_sExp_E9HAynn(int , float* , float* );
void fpk_vml_sExp_E9LAynn(int , float* , float* );
void fpk_vml_sExp_E9EPnnn(int , float* , float* );
void fpk_vml_dExp_E9HAynn(int , double* , double* );
void fpk_vml_dExp_E9LAynn(int , double* , double* );
void fpk_vml_dExp_E9EPnnn(int , double* , double* );

void fpk_vml_sExp_L9HAynn(int , float* , float* );
void fpk_vml_sExp_L9LAynn(int , float* , float* );
void fpk_vml_sExp_L9EPnnn(int , float* , float* );
void fpk_vml_dExp_L9HAynn(int , double* , double* );
void fpk_vml_dExp_L9LAynn(int , double* , double* );
void fpk_vml_dExp_L9EPnnn(int , double* , double* );

void fpk_vml_sExp_B3HAynn(int , float* , float* );
void fpk_vml_sExp_B3LAynn(int , float* , float* );
void fpk_vml_sExp_B3EPnnn(int , float* , float* );
void fpk_vml_dExp_B3HAynn(int , double* , double* );
void fpk_vml_dExp_B3LAynn(int , double* , double* );
void fpk_vml_dExp_B3EPnnn(int , double* , double* );

void fpk_vml_sExp_Z0HAynn(int , float* , float* );
void fpk_vml_sExp_Z0LAynn(int , float* , float* );
void fpk_vml_sExp_Z0EPnnn(int , float* , float* );
void fpk_vml_dExp_Z0HAynn(int , double* , double* );
void fpk_vml_dExp_Z0LAynn(int , double* , double* );
void fpk_vml_dExp_Z0EPnnn(int , double* , double* );


void fpk_vml_sErf_EXHAynn(int , float* , float* );
void fpk_vml_sErf_EXLAynn(int , float* , float* );
void fpk_vml_sErf_EXEPnnn(int , float* , float* );
void fpk_vml_dErf_EXHAynn(int , double* , double* );
void fpk_vml_dErf_EXLAynn(int , double* , double* );
void fpk_vml_dErf_EXEPnnn(int , double* , double* );

void fpk_vml_sErf_U8HAynn(int , float* , float* );
void fpk_vml_sErf_U8LAynn(int , float* , float* );
void fpk_vml_sErf_U8EPnnn(int , float* , float* );
void fpk_vml_dErf_U8HAynn(int , double* , double* );
void fpk_vml_dErf_U8LAynn(int , double* , double* );
void fpk_vml_dErf_U8EPnnn(int , double* , double* );

void fpk_vml_sErf_H8HAynn(int , float* , float* );
void fpk_vml_sErf_H8LAynn(int , float* , float* );
void fpk_vml_sErf_H8EPnnn(int , float* , float* );
void fpk_vml_dErf_H8HAynn(int , double* , double* );
void fpk_vml_dErf_H8LAynn(int , double* , double* );
void fpk_vml_dErf_H8EPnnn(int , double* , double* );

void fpk_vml_sErf_E9HAynn(int , float* , float* );
void fpk_vml_sErf_E9LAynn(int , float* , float* );
void fpk_vml_sErf_E9EPnnn(int , float* , float* );
void fpk_vml_dErf_E9HAynn(int , double* , double* );
void fpk_vml_dErf_E9LAynn(int , double* , double* );
void fpk_vml_dErf_E9EPnnn(int , double* , double* );

void fpk_vml_sErf_L9HAynn(int , float* , float* );
void fpk_vml_sErf_L9LAynn(int , float* , float* );
void fpk_vml_sErf_L9EPnnn(int , float* , float* );
void fpk_vml_dErf_L9HAynn(int , double* , double* );
void fpk_vml_dErf_L9LAynn(int , double* , double* );
void fpk_vml_dErf_L9EPnnn(int , double* , double* );

void fpk_vml_sErf_B3HAynn(int , float* , float* );
void fpk_vml_sErf_B3LAynn(int , float* , float* );
void fpk_vml_sErf_B3EPnnn(int , float* , float* );
void fpk_vml_dErf_B3HAynn(int , double* , double* );
void fpk_vml_dErf_B3LAynn(int , double* , double* );
void fpk_vml_dErf_B3EPnnn(int , double* , double* );

void fpk_vml_sErf_Z0HAynn(int , float* , float* );
void fpk_vml_sErf_Z0LAynn(int , float* , float* );
void fpk_vml_sErf_Z0EPnnn(int , float* , float* );
void fpk_vml_dErf_Z0HAynn(int , double* , double* );
void fpk_vml_dErf_Z0LAynn(int , double* , double* );
void fpk_vml_dErf_Z0EPnnn(int , double* , double* );


void fpk_vml_sErfInv_EXHAynn(int , float* , float* );
void fpk_vml_sErfInv_EXLAynn(int , float* , float* );
void fpk_vml_sErfInv_EXEPnnn(int , float* , float* );
void fpk_vml_dErfInv_EXHAynn(int , double* , double* );
void fpk_vml_dErfInv_EXLAynn(int , double* , double* );
void fpk_vml_dErfInv_EXEPnnn(int , double* , double* );

void fpk_vml_sErfInv_U8HAynn(int , float* , float* );
void fpk_vml_sErfInv_U8LAynn(int , float* , float* );
void fpk_vml_sErfInv_U8EPnnn(int , float* , float* );
void fpk_vml_dErfInv_U8HAynn(int , double* , double* );
void fpk_vml_dErfInv_U8LAynn(int , double* , double* );
void fpk_vml_dErfInv_U8EPnnn(int , double* , double* );

void fpk_vml_sErfInv_H8HAynn(int , float* , float* );
void fpk_vml_sErfInv_H8LAynn(int , float* , float* );
void fpk_vml_sErfInv_H8EPnnn(int , float* , float* );
void fpk_vml_dErfInv_H8HAynn(int , double* , double* );
void fpk_vml_dErfInv_H8LAynn(int , double* , double* );
void fpk_vml_dErfInv_H8EPnnn(int , double* , double* );

void fpk_vml_sErfInv_E9HAynn(int , float* , float* );
void fpk_vml_sErfInv_E9LAynn(int , float* , float* );
void fpk_vml_sErfInv_E9EPnnn(int , float* , float* );
void fpk_vml_dErfInv_E9HAynn(int , double* , double* );
void fpk_vml_dErfInv_E9LAynn(int , double* , double* );
void fpk_vml_dErfInv_E9EPnnn(int , double* , double* );

void fpk_vml_sErfInv_L9HAynn(int , float* , float* );
void fpk_vml_sErfInv_L9LAynn(int , float* , float* );
void fpk_vml_sErfInv_L9EPnnn(int , float* , float* );
void fpk_vml_dErfInv_L9HAynn(int , double* , double* );
void fpk_vml_dErfInv_L9LAynn(int , double* , double* );
void fpk_vml_dErfInv_L9EPnnn(int , double* , double* );

void fpk_vml_sErfInv_B3HAynn(int , float* , float* );
void fpk_vml_sErfInv_B3LAynn(int , float* , float* );
void fpk_vml_sErfInv_B3EPnnn(int , float* , float* );
void fpk_vml_dErfInv_B3HAynn(int , double* , double* );
void fpk_vml_dErfInv_B3LAynn(int , double* , double* );
void fpk_vml_dErfInv_B3EPnnn(int , double* , double* );

void fpk_vml_sErfInv_Z0HAynn(int , float* , float* );
void fpk_vml_sErfInv_Z0LAynn(int , float* , float* );
void fpk_vml_sErfInv_Z0EPnnn(int , float* , float* );
void fpk_vml_dErfInv_Z0HAynn(int , double* , double* );
void fpk_vml_dErfInv_Z0LAynn(int , double* , double* );
void fpk_vml_dErfInv_Z0EPnnn(int , double* , double* );


void fpk_vml_sCeil_EXHAynn(int , float* , float* );
void fpk_vml_sCeil_EXLAynn(int , float* , float* );
void fpk_vml_sCeil_EXEPnnn(int , float* , float* );
void fpk_vml_dCeil_EXHAynn(int , double* , double* );
void fpk_vml_dCeil_EXLAynn(int , double* , double* );
void fpk_vml_dCeil_EXEPnnn(int , double* , double* );

void fpk_vml_sCeil_U8HAynn(int , float* , float* );
void fpk_vml_sCeil_U8LAynn(int , float* , float* );
void fpk_vml_sCeil_U8EPnnn(int , float* , float* );
void fpk_vml_dCeil_U8HAynn(int , double* , double* );
void fpk_vml_dCeil_U8LAynn(int , double* , double* );
void fpk_vml_dCeil_U8EPnnn(int , double* , double* );

void fpk_vml_sCeil_H8HAynn(int , float* , float* );
void fpk_vml_sCeil_H8LAynn(int , float* , float* );
void fpk_vml_sCeil_H8EPnnn(int , float* , float* );
void fpk_vml_dCeil_H8HAynn(int , double* , double* );
void fpk_vml_dCeil_H8LAynn(int , double* , double* );
void fpk_vml_dCeil_H8EPnnn(int , double* , double* );

void fpk_vml_sCeil_E9HAynn(int , float* , float* );
void fpk_vml_sCeil_E9LAynn(int , float* , float* );
void fpk_vml_sCeil_E9EPnnn(int , float* , float* );
void fpk_vml_dCeil_E9HAynn(int , double* , double* );
void fpk_vml_dCeil_E9LAynn(int , double* , double* );
void fpk_vml_dCeil_E9EPnnn(int , double* , double* );

void fpk_vml_sCeil_L9HAynn(int , float* , float* );
void fpk_vml_sCeil_L9LAynn(int , float* , float* );
void fpk_vml_sCeil_L9EPnnn(int , float* , float* );
void fpk_vml_dCeil_L9HAynn(int , double* , double* );
void fpk_vml_dCeil_L9LAynn(int , double* , double* );
void fpk_vml_dCeil_L9EPnnn(int , double* , double* );

void fpk_vml_sCeil_B3HAynn(int , float* , float* );
void fpk_vml_sCeil_B3LAynn(int , float* , float* );
void fpk_vml_sCeil_B3EPnnn(int , float* , float* );
void fpk_vml_dCeil_B3HAynn(int , double* , double* );
void fpk_vml_dCeil_B3LAynn(int , double* , double* );
void fpk_vml_dCeil_B3EPnnn(int , double* , double* );

void fpk_vml_sCeil_Z0HAynn(int , float* , float* );
void fpk_vml_sCeil_Z0LAynn(int , float* , float* );
void fpk_vml_sCeil_Z0EPnnn(int , float* , float* );
void fpk_vml_dCeil_Z0HAynn(int , double* , double* );
void fpk_vml_dCeil_Z0LAynn(int , double* , double* );
void fpk_vml_dCeil_Z0EPnnn(int , double* , double* );


void fpk_vml_sPowx_EXHAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_EXLAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_EXEPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_EXHAynn(int , double* , double , double* );
void fpk_vml_dPowx_EXLAynn(int , double* , double , double* );
void fpk_vml_dPowx_EXEPnnn(int , double* , double , double* );

void fpk_vml_sPowx_U8HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_U8LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_U8EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_U8HAynn(int , double* , double , double* );
void fpk_vml_dPowx_U8LAynn(int , double* , double , double* );
void fpk_vml_dPowx_U8EPnnn(int , double* , double , double* );

void fpk_vml_sPowx_H8HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_H8LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_H8EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_H8HAynn(int , double* , double , double* );
void fpk_vml_dPowx_H8LAynn(int , double* , double , double* );
void fpk_vml_dPowx_H8EPnnn(int , double* , double , double* );

void fpk_vml_sPowx_E9HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_E9LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_E9EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_E9HAynn(int , double* , double , double* );
void fpk_vml_dPowx_E9LAynn(int , double* , double , double* );
void fpk_vml_dPowx_E9EPnnn(int , double* , double , double* );

void fpk_vml_sPowx_L9HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_L9LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_L9EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_L9HAynn(int , double* , double , double* );
void fpk_vml_dPowx_L9LAynn(int , double* , double , double* );
void fpk_vml_dPowx_L9EPnnn(int , double* , double , double* );

void fpk_vml_sPowx_B3HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_B3LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_B3EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_B3HAynn(int , double* , double , double* );
void fpk_vml_dPowx_B3LAynn(int , double* , double , double* );
void fpk_vml_dPowx_B3EPnnn(int , double* , double , double* );

void fpk_vml_sPowx_Z0HAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_Z0LAynn(int , float*  , float  , float* );
void fpk_vml_sPowx_Z0EPnnn(int , float*  , float  , float* );
void fpk_vml_dPowx_Z0HAynn(int , double* , double , double* );
void fpk_vml_dPowx_Z0LAynn(int , double* , double , double* );
void fpk_vml_dPowx_Z0EPnnn(int , double* , double , double* );


void fpk_vml_sSqrt_EXHAynn(int , float* , float* );
void fpk_vml_sSqrt_EXLAynn(int , float* , float* );
void fpk_vml_sSqrt_EXEPnnn(int , float* , float* );
void fpk_vml_dSqrt_EXHAynn(int , double* , double* );
void fpk_vml_dSqrt_EXLAynn(int , double* , double* );
void fpk_vml_dSqrt_EXEPnnn(int , double* , double* );

void fpk_vml_sSqrt_U8HAynn(int , float* , float* );
void fpk_vml_sSqrt_U8LAynn(int , float* , float* );
void fpk_vml_sSqrt_U8EPnnn(int , float* , float* );
void fpk_vml_dSqrt_U8HAynn(int , double* , double* );
void fpk_vml_dSqrt_U8LAynn(int , double* , double* );
void fpk_vml_dSqrt_U8EPnnn(int , double* , double* );

void fpk_vml_sSqrt_H8HAynn(int , float* , float* );
void fpk_vml_sSqrt_H8LAynn(int , float* , float* );
void fpk_vml_sSqrt_H8EPnnn(int , float* , float* );
void fpk_vml_dSqrt_H8HAynn(int , double* , double* );
void fpk_vml_dSqrt_H8LAynn(int , double* , double* );
void fpk_vml_dSqrt_H8EPnnn(int , double* , double* );

void fpk_vml_sSqrt_E9HAynn(int , float* , float* );
void fpk_vml_sSqrt_E9LAynn(int , float* , float* );
void fpk_vml_sSqrt_E9EPnnn(int , float* , float* );
void fpk_vml_dSqrt_E9HAynn(int , double* , double* );
void fpk_vml_dSqrt_E9LAynn(int , double* , double* );
void fpk_vml_dSqrt_E9EPnnn(int , double* , double* );

void fpk_vml_sSqrt_L9HAynn(int , float* , float* );
void fpk_vml_sSqrt_L9LAynn(int , float* , float* );
void fpk_vml_sSqrt_L9EPnnn(int , float* , float* );
void fpk_vml_dSqrt_L9HAynn(int , double* , double* );
void fpk_vml_dSqrt_L9LAynn(int , double* , double* );
void fpk_vml_dSqrt_L9EPnnn(int , double* , double* );

void fpk_vml_sSqrt_B3HAynn(int , float* , float* );
void fpk_vml_sSqrt_B3LAynn(int , float* , float* );
void fpk_vml_sSqrt_B3EPnnn(int , float* , float* );
void fpk_vml_dSqrt_B3HAynn(int , double* , double* );
void fpk_vml_dSqrt_B3LAynn(int , double* , double* );
void fpk_vml_dSqrt_B3EPnnn(int , double* , double* );

void fpk_vml_sSqrt_Z0HAynn(int , float* , float* );
void fpk_vml_sSqrt_Z0LAynn(int , float* , float* );
void fpk_vml_sSqrt_Z0EPnnn(int , float* , float* );
void fpk_vml_dSqrt_Z0HAynn(int , double* , double* );
void fpk_vml_dSqrt_Z0LAynn(int , double* , double* );
void fpk_vml_dSqrt_Z0EPnnn(int , double* , double* );


void fpk_vml_sTanh_EXHAynn(int , float* , float* );
void fpk_vml_sTanh_EXLAynn(int , float* , float* );
void fpk_vml_sTanh_EXEPnnn(int , float* , float* );
void fpk_vml_dTanh_EXHAynn(int , double* , double* );
void fpk_vml_dTanh_EXLAynn(int , double* , double* );
void fpk_vml_dTanh_EXEPnnn(int , double* , double* );

void fpk_vml_sTanh_U8HAynn(int , float* , float* );
void fpk_vml_sTanh_U8LAynn(int , float* , float* );
void fpk_vml_sTanh_U8EPnnn(int , float* , float* );
void fpk_vml_dTanh_U8HAynn(int , double* , double* );
void fpk_vml_dTanh_U8LAynn(int , double* , double* );
void fpk_vml_dTanh_U8EPnnn(int , double* , double* );

void fpk_vml_sTanh_H8HAynn(int , float* , float* );
void fpk_vml_sTanh_H8LAynn(int , float* , float* );
void fpk_vml_sTanh_H8EPnnn(int , float* , float* );
void fpk_vml_dTanh_H8HAynn(int , double* , double* );
void fpk_vml_dTanh_H8LAynn(int , double* , double* );
void fpk_vml_dTanh_H8EPnnn(int , double* , double* );

void fpk_vml_sTanh_E9HAynn(int , float* , float* );
void fpk_vml_sTanh_E9LAynn(int , float* , float* );
void fpk_vml_sTanh_E9EPnnn(int , float* , float* );
void fpk_vml_dTanh_E9HAynn(int , double* , double* );
void fpk_vml_dTanh_E9LAynn(int , double* , double* );
void fpk_vml_dTanh_E9EPnnn(int , double* , double* );

void fpk_vml_sTanh_L9HAynn(int , float* , float* );
void fpk_vml_sTanh_L9LAynn(int , float* , float* );
void fpk_vml_sTanh_L9EPnnn(int , float* , float* );
void fpk_vml_dTanh_L9HAynn(int , double* , double* );
void fpk_vml_dTanh_L9LAynn(int , double* , double* );
void fpk_vml_dTanh_L9EPnnn(int , double* , double* );

void fpk_vml_sTanh_B3HAynn(int , float* , float* );
void fpk_vml_sTanh_B3LAynn(int , float* , float* );
void fpk_vml_sTanh_B3EPnnn(int , float* , float* );
void fpk_vml_dTanh_B3HAynn(int , double* , double* );
void fpk_vml_dTanh_B3LAynn(int , double* , double* );
void fpk_vml_dTanh_B3EPnnn(int , double* , double* );

void fpk_vml_sTanh_Z0HAynn(int , float* , float* );
void fpk_vml_sTanh_Z0LAynn(int , float* , float* );
void fpk_vml_sTanh_Z0EPnnn(int , float* , float* );
void fpk_vml_dTanh_Z0HAynn(int , double* , double* );
void fpk_vml_dTanh_Z0LAynn(int , double* , double* );
void fpk_vml_dTanh_Z0EPnnn(int , double* , double* );


void fpk_vml_sLog1p_EXHAynn(int , float* , float* );
void fpk_vml_sLog1p_EXLAynn(int , float* , float* );
void fpk_vml_sLog1p_EXEPnnn(int , float* , float* );
void fpk_vml_dLog1p_EXHAynn(int , double* , double* );
void fpk_vml_dLog1p_EXLAynn(int , double* , double* );
void fpk_vml_dLog1p_EXEPnnn(int , double* , double* );

void fpk_vml_sLog1p_U8HAynn(int , float* , float* );
void fpk_vml_sLog1p_U8LAynn(int , float* , float* );
void fpk_vml_sLog1p_U8EPnnn(int , float* , float* );
void fpk_vml_dLog1p_U8HAynn(int , double* , double* );
void fpk_vml_dLog1p_U8LAynn(int , double* , double* );
void fpk_vml_dLog1p_U8EPnnn(int , double* , double* );

void fpk_vml_sLog1p_H8HAynn(int , float* , float* );
void fpk_vml_sLog1p_H8LAynn(int , float* , float* );
void fpk_vml_sLog1p_H8EPnnn(int , float* , float* );
void fpk_vml_dLog1p_H8HAynn(int , double* , double* );
void fpk_vml_dLog1p_H8LAynn(int , double* , double* );
void fpk_vml_dLog1p_H8EPnnn(int , double* , double* );

void fpk_vml_sLog1p_E9HAynn(int , float* , float* );
void fpk_vml_sLog1p_E9LAynn(int , float* , float* );
void fpk_vml_sLog1p_E9EPnnn(int , float* , float* );
void fpk_vml_dLog1p_E9HAynn(int , double* , double* );
void fpk_vml_dLog1p_E9LAynn(int , double* , double* );
void fpk_vml_dLog1p_E9EPnnn(int , double* , double* );

void fpk_vml_sLog1p_L9HAynn(int , float* , float* );
void fpk_vml_sLog1p_L9LAynn(int , float* , float* );
void fpk_vml_sLog1p_L9EPnnn(int , float* , float* );
void fpk_vml_dLog1p_L9HAynn(int , double* , double* );
void fpk_vml_dLog1p_L9LAynn(int , double* , double* );
void fpk_vml_dLog1p_L9EPnnn(int , double* , double* );

void fpk_vml_sLog1p_B3HAynn(int , float* , float* );
void fpk_vml_sLog1p_B3LAynn(int , float* , float* );
void fpk_vml_sLog1p_B3EPnnn(int , float* , float* );
void fpk_vml_dLog1p_B3HAynn(int , double* , double* );
void fpk_vml_dLog1p_B3LAynn(int , double* , double* );
void fpk_vml_dLog1p_B3EPnnn(int , double* , double* );

void fpk_vml_sLog1p_Z0HAynn(int , float* , float* );
void fpk_vml_sLog1p_Z0LAynn(int , float* , float* );
void fpk_vml_sLog1p_Z0EPnnn(int , float* , float* );
void fpk_vml_dLog1p_Z0HAynn(int , double* , double* );
void fpk_vml_dLog1p_Z0LAynn(int , double* , double* );
void fpk_vml_dLog1p_Z0EPnnn(int , double* , double* );


void fpk_vml_sCdfNormInv_EXHAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_EXLAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_EXEPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_EXHAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_EXLAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_EXEPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_U8HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_U8LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_U8EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_U8HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_U8LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_U8EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_H8HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_H8LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_H8EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_H8HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_H8LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_H8EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_E9HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_E9LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_E9EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_E9HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_E9LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_E9EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_L9HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_L9LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_L9EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_L9HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_L9LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_L9EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_B3HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_B3LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_B3EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_B3HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_B3LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_B3EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_Z0HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_Z0LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_Z0EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_Z0HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_Z0LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_Z0EPnnn(int , double* , double* );


int fpk_vsl_sub_kernel_ex_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_ex_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_ex_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_ex_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_ex_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_ex_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_ex_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_ex_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_ex_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_ex_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_ex_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_ex_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_ex_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_u8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_u8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_u8_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_u8_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_u8_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_u8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_u8_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_u8_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_u8_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_u8_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_u8_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_u8_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_u8_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_h8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_h8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_h8_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_h8_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_h8_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_h8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_h8_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_h8_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_h8_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_h8_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_h8_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_h8_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_h8_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_e9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_e9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_e9_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_e9_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_e9_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_e9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_e9_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_e9_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_e9_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_e9_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_e9_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_e9_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_e9_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_l9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_l9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_l9_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_l9_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_l9_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_l9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_l9_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_l9_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_l9_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_l9_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_l9_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_l9_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_l9_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_b3_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_b3_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_b3_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_b3_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_b3_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_b3_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_b3_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_b3_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_b3_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_b3_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_b3_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_b3_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_b3_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_z0_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_z0_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_z0_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_z0_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_z0_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_z0_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_z0_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_z0_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_z0_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_z0_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_z0_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_z0_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_z0_vslDeleteStream(void *);

#else

void fpk_vml_sLn_W7HAynn(int , float* , float* );
void fpk_vml_sLn_W7LAynn(int , float* , float* );
void fpk_vml_sLn_W7EPnnn(int , float* , float* );
void fpk_vml_dLn_W7HAynn(int , double* , double* );
void fpk_vml_dLn_W7LAynn(int , double* , double* );
void fpk_vml_dLn_W7EPnnn(int , double* , double* );

void fpk_vml_sLn_V8HAynn(int , float* , float* );
void fpk_vml_sLn_V8LAynn(int , float* , float* );
void fpk_vml_sLn_V8EPnnn(int , float* , float* );
void fpk_vml_dLn_V8HAynn(int , double* , double* );
void fpk_vml_dLn_V8LAynn(int , double* , double* );
void fpk_vml_dLn_V8EPnnn(int , double* , double* );

void fpk_vml_sLn_N8HAynn(int , float* , float* );
void fpk_vml_sLn_N8LAynn(int , float* , float* );
void fpk_vml_sLn_N8EPnnn(int , float* , float* );
void fpk_vml_dLn_N8HAynn(int , double* , double* );
void fpk_vml_dLn_N8LAynn(int , double* , double* );
void fpk_vml_dLn_N8EPnnn(int , double* , double* );

void fpk_vml_sLn_G9HAynn(int , float* , float* );
void fpk_vml_sLn_G9LAynn(int , float* , float* );
void fpk_vml_sLn_G9EPnnn(int , float* , float* );
void fpk_vml_dLn_G9HAynn(int , double* , double* );
void fpk_vml_dLn_G9LAynn(int , double* , double* );
void fpk_vml_dLn_G9EPnnn(int , double* , double* );

void fpk_vml_sLn_S9HAynn(int , float* , float* );
void fpk_vml_sLn_S9LAynn(int , float* , float* );
void fpk_vml_sLn_S9EPnnn(int , float* , float* );
void fpk_vml_dLn_S9HAynn(int , double* , double* );
void fpk_vml_dLn_S9LAynn(int , double* , double* );
void fpk_vml_dLn_S9EPnnn(int , double* , double* );

void fpk_vml_sLn_A3HAynn(int , float* , float* );
void fpk_vml_sLn_A3LAynn(int , float* , float* );
void fpk_vml_sLn_A3EPnnn(int , float* , float* );
void fpk_vml_dLn_A3HAynn(int , double* , double* );
void fpk_vml_dLn_A3LAynn(int , double* , double* );
void fpk_vml_dLn_A3EPnnn(int , double* , double* );

void fpk_vml_sLn_X0HAynn(int , float* , float* );
void fpk_vml_sLn_X0LAynn(int , float* , float* );
void fpk_vml_sLn_X0EPnnn(int , float* , float* );
void fpk_vml_dLn_X0HAynn(int , double* , double* );
void fpk_vml_dLn_X0LAynn(int , double* , double* );
void fpk_vml_dLn_X0EPnnn(int , double* , double* );


void fpk_vml_sExp_W7HAynn(int , float* , float* );
void fpk_vml_sExp_W7LAynn(int , float* , float* );
void fpk_vml_sExp_W7EPnnn(int , float* , float* );
void fpk_vml_dExp_W7HAynn(int , double* , double* );
void fpk_vml_dExp_W7LAynn(int , double* , double* );
void fpk_vml_dExp_W7EPnnn(int , double* , double* );

void fpk_vml_sExp_V8HAynn(int , float* , float* );
void fpk_vml_sExp_V8LAynn(int , float* , float* );
void fpk_vml_sExp_V8EPnnn(int , float* , float* );
void fpk_vml_dExp_V8HAynn(int , double* , double* );
void fpk_vml_dExp_V8LAynn(int , double* , double* );
void fpk_vml_dExp_V8EPnnn(int , double* , double* );

void fpk_vml_sExp_N8HAynn(int , float* , float* );
void fpk_vml_sExp_N8LAynn(int , float* , float* );
void fpk_vml_sExp_N8EPnnn(int , float* , float* );
void fpk_vml_dExp_N8HAynn(int , double* , double* );
void fpk_vml_dExp_N8LAynn(int , double* , double* );
void fpk_vml_dExp_N8EPnnn(int , double* , double* );

void fpk_vml_sExp_G9HAynn(int , float* , float* );
void fpk_vml_sExp_G9LAynn(int , float* , float* );
void fpk_vml_sExp_G9EPnnn(int , float* , float* );
void fpk_vml_dExp_G9HAynn(int , double* , double* );
void fpk_vml_dExp_G9LAynn(int , double* , double* );
void fpk_vml_dExp_G9EPnnn(int , double* , double* );

void fpk_vml_sExp_S9HAynn(int , float* , float* );
void fpk_vml_sExp_S9LAynn(int , float* , float* );
void fpk_vml_sExp_S9EPnnn(int , float* , float* );
void fpk_vml_dExp_S9HAynn(int , double* , double* );
void fpk_vml_dExp_S9LAynn(int , double* , double* );
void fpk_vml_dExp_S9EPnnn(int , double* , double* );

void fpk_vml_sExp_A3HAynn(int , float* , float* );
void fpk_vml_sExp_A3LAynn(int , float* , float* );
void fpk_vml_sExp_A3EPnnn(int , float* , float* );
void fpk_vml_dExp_A3HAynn(int , double* , double* );
void fpk_vml_dExp_A3LAynn(int , double* , double* );
void fpk_vml_dExp_A3EPnnn(int , double* , double* );

void fpk_vml_sExp_X0HAynn(int , float* , float* );
void fpk_vml_sExp_X0LAynn(int , float* , float* );
void fpk_vml_sExp_X0EPnnn(int , float* , float* );
void fpk_vml_dExp_X0HAynn(int , double* , double* );
void fpk_vml_dExp_X0LAynn(int , double* , double* );
void fpk_vml_dExp_X0EPnnn(int , double* , double* );


void fpk_vml_sErf_W7HAynn(int , float* , float* );
void fpk_vml_sErf_W7LAynn(int , float* , float* );
void fpk_vml_sErf_W7EPnnn(int , float* , float* );
void fpk_vml_dErf_W7HAynn(int , double* , double* );
void fpk_vml_dErf_W7LAynn(int , double* , double* );
void fpk_vml_dErf_W7EPnnn(int , double* , double* );

void fpk_vml_sErf_V8HAynn(int , float* , float* );
void fpk_vml_sErf_V8LAynn(int , float* , float* );
void fpk_vml_sErf_V8EPnnn(int , float* , float* );
void fpk_vml_dErf_V8HAynn(int , double* , double* );
void fpk_vml_dErf_V8LAynn(int , double* , double* );
void fpk_vml_dErf_V8EPnnn(int , double* , double* );

void fpk_vml_sErf_N8HAynn(int , float* , float* );
void fpk_vml_sErf_N8LAynn(int , float* , float* );
void fpk_vml_sErf_N8EPnnn(int , float* , float* );
void fpk_vml_dErf_N8HAynn(int , double* , double* );
void fpk_vml_dErf_N8LAynn(int , double* , double* );
void fpk_vml_dErf_N8EPnnn(int , double* , double* );

void fpk_vml_sErf_G9HAynn(int , float* , float* );
void fpk_vml_sErf_G9LAynn(int , float* , float* );
void fpk_vml_sErf_G9EPnnn(int , float* , float* );
void fpk_vml_dErf_G9HAynn(int , double* , double* );
void fpk_vml_dErf_G9LAynn(int , double* , double* );
void fpk_vml_dErf_G9EPnnn(int , double* , double* );

void fpk_vml_sErf_S9HAynn(int , float* , float* );
void fpk_vml_sErf_S9LAynn(int , float* , float* );
void fpk_vml_sErf_S9EPnnn(int , float* , float* );
void fpk_vml_dErf_S9HAynn(int , double* , double* );
void fpk_vml_dErf_S9LAynn(int , double* , double* );
void fpk_vml_dErf_S9EPnnn(int , double* , double* );

void fpk_vml_sErf_A3HAynn(int , float* , float* );
void fpk_vml_sErf_A3LAynn(int , float* , float* );
void fpk_vml_sErf_A3EPnnn(int , float* , float* );
void fpk_vml_dErf_A3HAynn(int , double* , double* );
void fpk_vml_dErf_A3LAynn(int , double* , double* );
void fpk_vml_dErf_A3EPnnn(int , double* , double* );

void fpk_vml_sErf_X0HAynn(int , float* , float* );
void fpk_vml_sErf_X0LAynn(int , float* , float* );
void fpk_vml_sErf_X0EPnnn(int , float* , float* );
void fpk_vml_dErf_X0HAynn(int , double* , double* );
void fpk_vml_dErf_X0LAynn(int , double* , double* );
void fpk_vml_dErf_X0EPnnn(int , double* , double* );


void fpk_vml_sErfInv_W7HAynn(int , float* , float* );
void fpk_vml_sErfInv_W7LAynn(int , float* , float* );
void fpk_vml_sErfInv_W7EPnnn(int , float* , float* );
void fpk_vml_dErfInv_W7HAynn(int , double* , double* );
void fpk_vml_dErfInv_W7LAynn(int , double* , double* );
void fpk_vml_dErfInv_W7EPnnn(int , double* , double* );

void fpk_vml_sErfInv_V8HAynn(int , float* , float* );
void fpk_vml_sErfInv_V8LAynn(int , float* , float* );
void fpk_vml_sErfInv_V8EPnnn(int , float* , float* );
void fpk_vml_dErfInv_V8HAynn(int , double* , double* );
void fpk_vml_dErfInv_V8LAynn(int , double* , double* );
void fpk_vml_dErfInv_V8EPnnn(int , double* , double* );

void fpk_vml_sErfInv_N8HAynn(int , float* , float* );
void fpk_vml_sErfInv_N8LAynn(int , float* , float* );
void fpk_vml_sErfInv_N8EPnnn(int , float* , float* );
void fpk_vml_dErfInv_N8HAynn(int , double* , double* );
void fpk_vml_dErfInv_N8LAynn(int , double* , double* );
void fpk_vml_dErfInv_N8EPnnn(int , double* , double* );

void fpk_vml_sErfInv_G9HAynn(int , float* , float* );
void fpk_vml_sErfInv_G9LAynn(int , float* , float* );
void fpk_vml_sErfInv_G9EPnnn(int , float* , float* );
void fpk_vml_dErfInv_G9HAynn(int , double* , double* );
void fpk_vml_dErfInv_G9LAynn(int , double* , double* );
void fpk_vml_dErfInv_G9EPnnn(int , double* , double* );

void fpk_vml_sErfInv_S9HAynn(int , float* , float* );
void fpk_vml_sErfInv_S9LAynn(int , float* , float* );
void fpk_vml_sErfInv_S9EPnnn(int , float* , float* );
void fpk_vml_dErfInv_S9HAynn(int , double* , double* );
void fpk_vml_dErfInv_S9LAynn(int , double* , double* );
void fpk_vml_dErfInv_S9EPnnn(int , double* , double* );

void fpk_vml_sErfInv_A3HAynn(int , float* , float* );
void fpk_vml_sErfInv_A3LAynn(int , float* , float* );
void fpk_vml_sErfInv_A3EPnnn(int , float* , float* );
void fpk_vml_dErfInv_A3HAynn(int , double* , double* );
void fpk_vml_dErfInv_A3LAynn(int , double* , double* );
void fpk_vml_dErfInv_A3EPnnn(int , double* , double* );

void fpk_vml_sErfInv_X0HAynn(int , float* , float* );
void fpk_vml_sErfInv_X0LAynn(int , float* , float* );
void fpk_vml_sErfInv_X0EPnnn(int , float* , float* );
void fpk_vml_dErfInv_X0HAynn(int , double* , double* );
void fpk_vml_dErfInv_X0LAynn(int , double* , double* );
void fpk_vml_dErfInv_X0EPnnn(int , double* , double* );


void fpk_vml_sCeil_W7HAynn(int , float* , float* );
void fpk_vml_sCeil_W7LAynn(int , float* , float* );
void fpk_vml_sCeil_W7EPnnn(int , float* , float* );
void fpk_vml_dCeil_W7HAynn(int , double* , double* );
void fpk_vml_dCeil_W7LAynn(int , double* , double* );
void fpk_vml_dCeil_W7EPnnn(int , double* , double* );

void fpk_vml_sCeil_V8HAynn(int , float* , float* );
void fpk_vml_sCeil_V8LAynn(int , float* , float* );
void fpk_vml_sCeil_V8EPnnn(int , float* , float* );
void fpk_vml_dCeil_V8HAynn(int , double* , double* );
void fpk_vml_dCeil_V8LAynn(int , double* , double* );
void fpk_vml_dCeil_V8EPnnn(int , double* , double* );

void fpk_vml_sCeil_N8HAynn(int , float* , float* );
void fpk_vml_sCeil_N8LAynn(int , float* , float* );
void fpk_vml_sCeil_N8EPnnn(int , float* , float* );
void fpk_vml_dCeil_N8HAynn(int , double* , double* );
void fpk_vml_dCeil_N8LAynn(int , double* , double* );
void fpk_vml_dCeil_N8EPnnn(int , double* , double* );

void fpk_vml_sCeil_G9HAynn(int , float* , float* );
void fpk_vml_sCeil_G9LAynn(int , float* , float* );
void fpk_vml_sCeil_G9EPnnn(int , float* , float* );
void fpk_vml_dCeil_G9HAynn(int , double* , double* );
void fpk_vml_dCeil_G9LAynn(int , double* , double* );
void fpk_vml_dCeil_G9EPnnn(int , double* , double* );

void fpk_vml_sCeil_S9HAynn(int , float* , float* );
void fpk_vml_sCeil_S9LAynn(int , float* , float* );
void fpk_vml_sCeil_S9EPnnn(int , float* , float* );
void fpk_vml_dCeil_S9HAynn(int , double* , double* );
void fpk_vml_dCeil_S9LAynn(int , double* , double* );
void fpk_vml_dCeil_S9EPnnn(int , double* , double* );

void fpk_vml_sCeil_A3HAynn(int , float* , float* );
void fpk_vml_sCeil_A3LAynn(int , float* , float* );
void fpk_vml_sCeil_A3EPnnn(int , float* , float* );
void fpk_vml_dCeil_A3HAynn(int , double* , double* );
void fpk_vml_dCeil_A3LAynn(int , double* , double* );
void fpk_vml_dCeil_A3EPnnn(int , double* , double* );

void fpk_vml_sCeil_X0HAynn(int , float* , float* );
void fpk_vml_sCeil_X0LAynn(int , float* , float* );
void fpk_vml_sCeil_X0EPnnn(int , float* , float* );
void fpk_vml_dCeil_X0HAynn(int , double* , double* );
void fpk_vml_dCeil_X0LAynn(int , double* , double* );
void fpk_vml_dCeil_X0EPnnn(int , double* , double* );


void fpk_vml_sPowx_W7HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_W7LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_W7EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_W7HAynn(int , double*, double , double* );
void fpk_vml_dPowx_W7LAynn(int , double*, double , double* );
void fpk_vml_dPowx_W7EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_V8HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_V8LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_V8EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_V8HAynn(int , double*, double , double* );
void fpk_vml_dPowx_V8LAynn(int , double*, double , double* );
void fpk_vml_dPowx_V8EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_N8HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_N8LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_N8EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_N8HAynn(int , double*, double , double* );
void fpk_vml_dPowx_N8LAynn(int , double*, double , double* );
void fpk_vml_dPowx_N8EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_G9HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_G9LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_G9EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_G9HAynn(int , double*, double , double* );
void fpk_vml_dPowx_G9LAynn(int , double*, double , double* );
void fpk_vml_dPowx_G9EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_S9HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_S9LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_S9EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_S9HAynn(int , double*, double , double* );
void fpk_vml_dPowx_S9LAynn(int , double*, double , double* );
void fpk_vml_dPowx_S9EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_A3HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_A3LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_A3EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_A3HAynn(int , double*, double , double* );
void fpk_vml_dPowx_A3LAynn(int , double*, double , double* );
void fpk_vml_dPowx_A3EPnnn(int , double*, double , double* );

void fpk_vml_sPowx_X0HAynn(int , float* , float  , float* );
void fpk_vml_sPowx_X0LAynn(int , float* , float  , float* );
void fpk_vml_sPowx_X0EPnnn(int , float* , float  , float* );
void fpk_vml_dPowx_X0HAynn(int , double*, double , double* );
void fpk_vml_dPowx_X0LAynn(int , double*, double , double* );
void fpk_vml_dPowx_X0EPnnn(int , double*, double , double* );


void fpk_vml_sSqrt_W7HAynn(int , float* , float* );
void fpk_vml_sSqrt_W7LAynn(int , float* , float* );
void fpk_vml_sSqrt_W7EPnnn(int , float* , float* );
void fpk_vml_dSqrt_W7HAynn(int , double* , double* );
void fpk_vml_dSqrt_W7LAynn(int , double* , double* );
void fpk_vml_dSqrt_W7EPnnn(int , double* , double* );

void fpk_vml_sSqrt_V8HAynn(int , float* , float* );
void fpk_vml_sSqrt_V8LAynn(int , float* , float* );
void fpk_vml_sSqrt_V8EPnnn(int , float* , float* );
void fpk_vml_dSqrt_V8HAynn(int , double* , double* );
void fpk_vml_dSqrt_V8LAynn(int , double* , double* );
void fpk_vml_dSqrt_V8EPnnn(int , double* , double* );

void fpk_vml_sSqrt_N8HAynn(int , float* , float* );
void fpk_vml_sSqrt_N8LAynn(int , float* , float* );
void fpk_vml_sSqrt_N8EPnnn(int , float* , float* );
void fpk_vml_dSqrt_N8HAynn(int , double* , double* );
void fpk_vml_dSqrt_N8LAynn(int , double* , double* );
void fpk_vml_dSqrt_N8EPnnn(int , double* , double* );

void fpk_vml_sSqrt_G9HAynn(int , float* , float* );
void fpk_vml_sSqrt_G9LAynn(int , float* , float* );
void fpk_vml_sSqrt_G9EPnnn(int , float* , float* );
void fpk_vml_dSqrt_G9HAynn(int , double* , double* );
void fpk_vml_dSqrt_G9LAynn(int , double* , double* );
void fpk_vml_dSqrt_G9EPnnn(int , double* , double* );

void fpk_vml_sSqrt_S9HAynn(int , float* , float* );
void fpk_vml_sSqrt_S9LAynn(int , float* , float* );
void fpk_vml_sSqrt_S9EPnnn(int , float* , float* );
void fpk_vml_dSqrt_S9HAynn(int , double* , double* );
void fpk_vml_dSqrt_S9LAynn(int , double* , double* );
void fpk_vml_dSqrt_S9EPnnn(int , double* , double* );

void fpk_vml_sSqrt_A3HAynn(int , float* , float* );
void fpk_vml_sSqrt_A3LAynn(int , float* , float* );
void fpk_vml_sSqrt_A3EPnnn(int , float* , float* );
void fpk_vml_dSqrt_A3HAynn(int , double* , double* );
void fpk_vml_dSqrt_A3LAynn(int , double* , double* );
void fpk_vml_dSqrt_A3EPnnn(int , double* , double* );

void fpk_vml_sSqrt_X0HAynn(int , float* , float* );
void fpk_vml_sSqrt_X0LAynn(int , float* , float* );
void fpk_vml_sSqrt_X0EPnnn(int , float* , float* );
void fpk_vml_dSqrt_X0HAynn(int , double* , double* );
void fpk_vml_dSqrt_X0LAynn(int , double* , double* );
void fpk_vml_dSqrt_X0EPnnn(int , double* , double* );


void fpk_vml_sTanh_W7HAynn(int , float* , float* );
void fpk_vml_sTanh_W7LAynn(int , float* , float* );
void fpk_vml_sTanh_W7EPnnn(int , float* , float* );
void fpk_vml_dTanh_W7HAynn(int , double* , double* );
void fpk_vml_dTanh_W7LAynn(int , double* , double* );
void fpk_vml_dTanh_W7EPnnn(int , double* , double* );

void fpk_vml_sTanh_V8HAynn(int , float* , float* );
void fpk_vml_sTanh_V8LAynn(int , float* , float* );
void fpk_vml_sTanh_V8EPnnn(int , float* , float* );
void fpk_vml_dTanh_V8HAynn(int , double* , double* );
void fpk_vml_dTanh_V8LAynn(int , double* , double* );
void fpk_vml_dTanh_V8EPnnn(int , double* , double* );

void fpk_vml_sTanh_N8HAynn(int , float* , float* );
void fpk_vml_sTanh_N8LAynn(int , float* , float* );
void fpk_vml_sTanh_N8EPnnn(int , float* , float* );
void fpk_vml_dTanh_N8HAynn(int , double* , double* );
void fpk_vml_dTanh_N8LAynn(int , double* , double* );
void fpk_vml_dTanh_N8EPnnn(int , double* , double* );

void fpk_vml_sTanh_G9HAynn(int , float* , float* );
void fpk_vml_sTanh_G9LAynn(int , float* , float* );
void fpk_vml_sTanh_G9EPnnn(int , float* , float* );
void fpk_vml_dTanh_G9HAynn(int , double* , double* );
void fpk_vml_dTanh_G9LAynn(int , double* , double* );
void fpk_vml_dTanh_G9EPnnn(int , double* , double* );

void fpk_vml_sTanh_S9HAynn(int , float* , float* );
void fpk_vml_sTanh_S9LAynn(int , float* , float* );
void fpk_vml_sTanh_S9EPnnn(int , float* , float* );
void fpk_vml_dTanh_S9HAynn(int , double* , double* );
void fpk_vml_dTanh_S9LAynn(int , double* , double* );
void fpk_vml_dTanh_S9EPnnn(int , double* , double* );

void fpk_vml_sTanh_A3HAynn(int , float* , float* );
void fpk_vml_sTanh_A3LAynn(int , float* , float* );
void fpk_vml_sTanh_A3EPnnn(int , float* , float* );
void fpk_vml_dTanh_A3HAynn(int , double* , double* );
void fpk_vml_dTanh_A3LAynn(int , double* , double* );
void fpk_vml_dTanh_A3EPnnn(int , double* , double* );

void fpk_vml_sTanh_X0HAynn(int , float* , float* );
void fpk_vml_sTanh_X0LAynn(int , float* , float* );
void fpk_vml_sTanh_X0EPnnn(int , float* , float* );
void fpk_vml_dTanh_X0HAynn(int , double* , double* );
void fpk_vml_dTanh_X0LAynn(int , double* , double* );
void fpk_vml_dTanh_X0EPnnn(int , double* , double* );


void fpk_vml_sLog1p_W7HAynn(int , float* , float* );
void fpk_vml_sLog1p_W7LAynn(int , float* , float* );
void fpk_vml_sLog1p_W7EPnnn(int , float* , float* );
void fpk_vml_dLog1p_W7HAynn(int , double* , double* );
void fpk_vml_dLog1p_W7LAynn(int , double* , double* );
void fpk_vml_dLog1p_W7EPnnn(int , double* , double* );

void fpk_vml_sLog1p_V8HAynn(int , float* , float* );
void fpk_vml_sLog1p_V8LAynn(int , float* , float* );
void fpk_vml_sLog1p_V8EPnnn(int , float* , float* );
void fpk_vml_dLog1p_V8HAynn(int , double* , double* );
void fpk_vml_dLog1p_V8LAynn(int , double* , double* );
void fpk_vml_dLog1p_V8EPnnn(int , double* , double* );

void fpk_vml_sLog1p_N8HAynn(int , float* , float* );
void fpk_vml_sLog1p_N8LAynn(int , float* , float* );
void fpk_vml_sLog1p_N8EPnnn(int , float* , float* );
void fpk_vml_dLog1p_N8HAynn(int , double* , double* );
void fpk_vml_dLog1p_N8LAynn(int , double* , double* );
void fpk_vml_dLog1p_N8EPnnn(int , double* , double* );

void fpk_vml_sLog1p_G9HAynn(int , float* , float* );
void fpk_vml_sLog1p_G9LAynn(int , float* , float* );
void fpk_vml_sLog1p_G9EPnnn(int , float* , float* );
void fpk_vml_dLog1p_G9HAynn(int , double* , double* );
void fpk_vml_dLog1p_G9LAynn(int , double* , double* );
void fpk_vml_dLog1p_G9EPnnn(int , double* , double* );

void fpk_vml_sLog1p_S9HAynn(int , float* , float* );
void fpk_vml_sLog1p_S9LAynn(int , float* , float* );
void fpk_vml_sLog1p_S9EPnnn(int , float* , float* );
void fpk_vml_dLog1p_S9HAynn(int , double* , double* );
void fpk_vml_dLog1p_S9LAynn(int , double* , double* );
void fpk_vml_dLog1p_S9EPnnn(int , double* , double* );

void fpk_vml_sLog1p_A3HAynn(int , float* , float* );
void fpk_vml_sLog1p_A3LAynn(int , float* , float* );
void fpk_vml_sLog1p_A3EPnnn(int , float* , float* );
void fpk_vml_dLog1p_A3HAynn(int , double* , double* );
void fpk_vml_dLog1p_A3LAynn(int , double* , double* );
void fpk_vml_dLog1p_A3EPnnn(int , double* , double* );

void fpk_vml_sLog1p_X0HAynn(int , float* , float* );
void fpk_vml_sLog1p_X0LAynn(int , float* , float* );
void fpk_vml_sLog1p_X0EPnnn(int , float* , float* );
void fpk_vml_dLog1p_X0HAynn(int , double* , double* );
void fpk_vml_dLog1p_X0LAynn(int , double* , double* );
void fpk_vml_dLog1p_X0EPnnn(int , double* , double* );


void fpk_vml_sCdfNormInv_W7HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_W7LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_W7EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_W7HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_W7LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_W7EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_V8HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_V8LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_V8EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_V8HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_V8LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_V8EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_N8HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_N8LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_N8EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_N8HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_N8LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_N8EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_G9HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_G9LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_G9EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_G9HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_G9LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_G9EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_S9HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_S9LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_S9EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_S9HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_S9LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_S9EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_A3HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_A3LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_A3EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_A3HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_A3LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_A3EPnnn(int , double* , double* );

void fpk_vml_sCdfNormInv_X0HAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_X0LAynn(int , float* , float* );
void fpk_vml_sCdfNormInv_X0EPnnn(int , float* , float* );
void fpk_vml_dCdfNormInv_X0HAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_X0LAynn(int , double* , double* );
void fpk_vml_dCdfNormInv_X0EPnnn(int , double* , double* );


int fpk_vsl_sub_kernel_w7_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_w7_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_w7_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_w7_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_w7_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_w7_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_w7_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_w7_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_w7_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_w7_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_w7_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_w7_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_w7_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_v8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_v8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_v8_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_v8_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_v8_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_v8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_v8_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_v8_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_v8_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_v8_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_v8_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_v8_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_v8_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_n8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_n8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_n8_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_n8_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_n8_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_n8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_n8_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_n8_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_n8_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_n8_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_n8_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_n8_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_n8_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_g9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_g9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_g9_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_g9_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_g9_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_g9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_g9_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_g9_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_g9_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_g9_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_g9_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_g9_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_g9_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_s9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_s9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_s9_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_s9_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_s9_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_s9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_s9_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_s9_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_s9_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_s9_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_s9_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_s9_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_s9_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_a3_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_a3_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_a3_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_a3_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_a3_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_a3_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_a3_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_a3_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_a3_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_a3_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_a3_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_a3_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_a3_vslDeleteStream(void *);

int fpk_vsl_sub_kernel_x0_vsldSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, double *, double *, __int64 *, int );
int fpk_vsl_sub_kernel_x0_vslsSSNewTask(DAAL_VSLSSTaskPtr *, __int64 *, __int64 *, __int64 *, float *, float *, __int64 *, int );
int fpk_vsl_sub_kernel_x0_vsldSSEditTask(DAAL_VSLSSTaskPtr , __int64 , double *);
int fpk_vsl_sub_kernel_x0_vslsSSEditTask(DAAL_VSLSSTaskPtr , __int64 , float *);
int fpk_vsl_sub_kernel_x0_vsliSSEditTask(DAAL_VSLSSTaskPtr , __int64 , __int64 *);
int fpk_vsl_sub_kernel_x0_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_x0_dSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSBasic(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSOutliersDetection(void* , __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_x0_vslsSSEditOutDetect(void *, __int64 *, float *, float *);
int fpk_vsl_sub_kernel_x0_vsldSSEditOutDetect(void *, __int64 *, double *, double *);
int fpk_vsl_kernel_x0_dSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSMahDistance(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSSort(void *, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSStreamQuantiles(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSMissingValues(void*, __int64 , __int64 , struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_iRngUniform(int , void * , int , int [] , int , int );
int fpk_vsl_kernel_x0_iRngBernoulli(int , void * , int , int [] , double );
int fpk_vsl_sub_kernel_x0_vslNewStreamEx(void *, int , int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_x0_vslDeleteStream(void *);

#endif


#if defined(__cplusplus)
}
#endif


#endif
