/* file: service_math_mkl.h */
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
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_MATH_MKL_H__
#define __SERVICE_MATH_MKL_H__


#include <math.h>
#include "vmlvsl.h"
#include "service_defines.h"


#if !defined(__DAAL_CONCAT5)
  #define __DAAL_CONCAT5(a,b,c,d,e) __DAAL_CONCAT51(a,b,c,d,e)
  #define __DAAL_CONCAT51(a,b,c,d,e) a##b##c##d##e
#endif


#define VMLFN(f_cpu,f_name,f_suff)   __DAAL_CONCAT5(fpk_vml_,f_name,_,f_cpu,f_suff)
#define VMLFN_CALL(f_name,f_suff,f_args)  VMLFN_CALL1(f_name,f_suff,f_args)


#if defined(_WIN64) || defined(__x86_64__)

#define VMLFN_CALL1(f_name,f_suff,f_args)                 \
    if(avx512 == cpu)                                     \
    {                                                     \
        VMLFN(Z0,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx512_mic == cpu)                                 \
    {                                                     \
        VMLFN(B3,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx2 == cpu)                                       \
    {                                                     \
        VMLFN(L9,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx == cpu)                                        \
    {                                                     \
        VMLFN(E9,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(sse42 == cpu)                                      \
    {                                                     \
        VMLFN(H8,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(ssse3 == cpu)                                      \
    {                                                     \
        VMLFN(U8,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(sse2 == cpu)                                       \
    {                                                     \
        VMLFN(EX,f_name,f_suff) f_args;                   \
        return;                                           \
    }

#else

#define VMLFN_CALL1(f_name,f_suff,f_args)                 \
    if(avx512 == cpu)                                     \
    {                                                     \
        VMLFN(X0,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx512_mic == cpu)                                 \
    {                                                     \
        VMLFN(A3,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx2 == cpu)                                       \
    {                                                     \
        VMLFN(S9,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(avx == cpu)                                        \
    {                                                     \
        VMLFN(G9,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(sse42 == cpu)                                      \
    {                                                     \
        VMLFN(N8,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(ssse3 == cpu)                                      \
    {                                                     \
        VMLFN(V8,f_name,f_suff) f_args;                   \
        return;                                           \
    }                                                     \
    if(sse2 == cpu)                                       \
    {                                                     \
        VMLFN(W7,f_name,f_suff) f_args;                   \
        return;                                           \
    }

#endif

namespace daal
{
namespace internal
{
namespace mkl
{

template<typename fpType, CpuType cpu>
struct MklMath {};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklMath<double, cpu>
{
    typedef size_t SizeType;

    static double sFabs(double in)
    {
        return ( in >= 0.0f ) ? in : -in;
    }

    static double sMin(double in1, double in2)
    {
        return ( in1 > in2 ) ? in2 : in1;
    }

    static double sMax(double in1, double in2)
    {
        return ( in1 < in2 ) ? in2 : in1;
    }

    static double sSqrt(double in)
    {
        return sqrt(in);
    }

    static double sPowx(double in, double in1)
    {
        double r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static double sCeil(double in)
    {
        double r;
        vCeil(1, &in, &r);
        return r;
    }

    static double sErfInv(double in)
    {
        double r;
        vErfInv(1, &in, &r);
        return r;
    }

    static double sLog(double in)
    {
        double r;
        vLog(1, &in, &r);
        return r;
    }

    static double sCdfNormInv(double in)
    {
        double r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, double *in, double in1, double *out)
    {
        VMLFN_CALL(dPowx,HAynn,((int)n, in, in1, out));
    }

    static void vCeil(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dCeil,HAynn,((int)n, in, out));
    }

    static void vErfInv(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dErfInv,HAynn,((int)n, in, out));
    }

    static void vErf(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dErf,HAynn,((int)n, in, out));
    }

    static void vExp(SizeType n, double* in, double* out)
    {
        VMLFN_CALL(dExp,HAynn,((int)n, in, out));
    }

    static void vTanh(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dTanh,HAynn,((int)n, in, out));
    }

    static void vSqrt(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dSqrt,HAynn,((int)n, in, out));
    }

    static void vLog(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dLn,HAynn,((int)n, in, out));
    }

    static void vLog1p(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dLog1p,HAynn,((int)n, in, out));
    }

    static void vCdfNormInv(SizeType n, double *in, double *out)
    {
        VMLFN_CALL(dCdfNormInv,HAynn,((int)n, in, out));
    }
};


/*
// Single precision functions definition
*/

template<CpuType cpu>
struct MklMath<float, cpu>
{
    typedef size_t SizeType;

    static float sFabs(float in)
    {
        return ( in >= 0.0 ) ? in : -in;
    }

    static float sMin(float in1, float in2)
    {
        return ( in1 > in2 ) ? in2 : in1;
    }

    static float sMax(float in1, float in2)
    {
        return ( in1 < in2 ) ? in2 : in1;
    }

    static float sSqrt(float in)
    {
        return sqrt(in);
    }

    static float sPowx(float in, float in1)
    {
        float r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static float sCeil(float in)
    {
        float r;
        vCeil(1, &in, &r);
        return r;
    }

    static float sErfInv(float in)
    {
        float r;
        vErfInv(1, &in, &r);
        return r;
    }

    static float sLog(float in)
    {
        float r;
        vLog(1, &in, &r);
        return r;
    }

    static float sCdfNormInv(float in)
    {
        float r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, float *in, float in1, float *out)
    {
        VMLFN_CALL(sPowx,HAynn,((int)n, in, in1, out));
    }

    static void vCeil(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sCeil,HAynn,((int)n, in, out));
    }

    static void vErfInv(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sErfInv,HAynn,((int)n, in, out));
    }

    static void vErf(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sErf,HAynn,((int)n, in, out));
    }

    static void vExp(SizeType n, float* in, float* out)
    {
        VMLFN_CALL(sExp,HAynn,((int)n, in, out));
    }

    static void vTanh(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sTanh,HAynn,((int)n, in, out));
    }

    static void vSqrt(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sSqrt,HAynn,((int)n, in, out));
    }

    static void vLog(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sLn,HAynn,((int)n, in, out));
    }

    static void vLog1p(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sLog1p,HAynn,((int)n, in, out));
    }

    static void vCdfNormInv(SizeType n, float *in, float *out)
    {
        VMLFN_CALL(sCdfNormInv,HAynn,((int)n, in, out));
    }
};

}
}
}

namespace daal
{

template<typename interm, CpuType cpu>
inline interm sFabs(interm in)
{
    return ( in >= (interm)0.0 ) ? in : -in;
}

template<typename interm, CpuType cpu>
inline interm sMin(interm in1, interm in2)
{
    return ( in1 > in2 ) ? in2 : in1;
}

template<typename interm, CpuType cpu>
inline interm sMax(interm in1, interm in2)
{
    return ( in1 > in2 ) ? in1 : in2;
}


/**
*** scalar Sqrt functions (sSqrt)
**/

template<CpuType cpu>
inline float sSqrt(float in)
{
    return sqrtf(in);
}

template<CpuType cpu>
inline double sSqrt(double in)
{
    return sqrt(in);
}

/**
*** vector and scalar Powx functions (vPowx/sPowx)
**/

template<CpuType cpu>
void vPowx(size_t n, float *in, float in1, float *out)
{
    VMLFN_CALL(sPowx,HAynn,((int)n, in, in1, out))
}

template<CpuType cpu>
void vPowx(size_t n, double *in, double in1, double *out)
{
    VMLFN_CALL(dPowx,HAynn,((int)n, in, in1, out))
}

template<CpuType cpu>
inline float sPowx(float in, float in1)
{
    float r;

    vPowx<cpu>(1,&in,in1,&r);

    return r;
}

template<CpuType cpu>
inline double sPowx(double in, double in1)
{
    double r;

    vPowx<cpu>(1,&in,in1,&r);

    return r;
}

/**
*** vector and scalar Ceil functions (vCeil/sCeil)
**/

template<CpuType cpu>
void vCeil(size_t n, float *in, float *out)
{
    VMLFN_CALL(sCeil,HAynn,((int)n, in, out))
}

template<CpuType cpu>
void vCeil(size_t n, double *in, double *out)
{
    VMLFN_CALL(dCeil,HAynn,((int)n, in, out))
}

template<CpuType cpu>
inline float sCeil(float in)
{
    float r;

    vCeil<cpu>(1,&in,&r);

    return r;
}

template<CpuType cpu>
inline double sCeil(double in)
{
    double r;

    vCeil<cpu>(1,&in,&r);

    return r;
}

/**
*** vector and scalar ErfInv functions (vErfInv/sErfInv)
**/

template<CpuType cpu>
void vErfInv(size_t n, float *in, float *out)
{
    VMLFN_CALL(sErfInv,HAynn,((int)n, in, out))
}

template<CpuType cpu>
void vErfInv(size_t n, double *in, double *out)
{
    VMLFN_CALL(dErfInv,HAynn,((int)n, in, out))
}

template<CpuType cpu>
inline float sErfInv(float in)
{
    float r;

    vErfInv<cpu>(1,&in,&r);

    return r;
}

template<CpuType cpu>
inline double sErfInv(double in)
{
    double r;

    vErfInv<cpu>(1,&in,&r);

    return r;
}


/**
*** vector Erf functions (vErf)
**/

template<CpuType cpu>
void vErf(size_t n, float *in, float *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for (size_t i = 0; i < n; i++)
    {
        out[i] = erff(in[i]);
    }
#else
    VMLFN_CALL(sErf,HAynn,((int)n, in, out))
#endif
}

template<CpuType cpu>
void vErf(size_t n, double *in, double *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for (size_t i = 0; i < n; i++)
    {
        out[i] = erf(in[i]);
    }
#else
    VMLFN_CALL(dErf,HAynn,((int)n, in, out))
#endif
}


/**
*** vector Exp functions (vExp)
**/

template<CpuType cpu>
void vExp(size_t n, float* in, float* out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = expf(in[i]);
    }
#else
    VMLFN_CALL(sExp,HAynn,((int)n, in, out))
#endif
}

template<CpuType cpu>
void vExp(size_t n, double* in, double* out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = exp(in[i]);
    }
#else
    VMLFN_CALL(dExp,HAynn,((int)n, in, out))
#endif
}


/**
*** vector Tanh functions (vTanh)
**/

template<CpuType cpu>
void vTanh(size_t n, float *in, float *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = tanhf(in[i]);
    }
#else
    VMLFN_CALL(sTanh, HAynn, ((int)n, in, out))
#endif
}

template<CpuType cpu>
void vTanh(size_t n, double *in, double *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = tanh(in[i]);
    }
#else
    VMLFN_CALL(dTanh, HAynn, ((int)n, in, out))
#endif
}

/**
*** vector Sqrt functions (vSqrt)
**/

template<CpuType cpu>
void vSqrt(size_t n, float *in, float *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = sqrtf(in[i]);
    }
#else
    VMLFN_CALL(sSqrt, HAynn, ((int)n, in, out))
#endif
}

template<CpuType cpu>
void vSqrt(size_t n, double *in, double *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = sqrt(in[i]);
    }
#else
    VMLFN_CALL(dSqrt, HAynn, ((int)n, in, out))
#endif
}

/**
*** vector and scalar Log functions (vLog / sLog)
**/

template<CpuType cpu>
void vLog(size_t n, float *in, float *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = logf(in[i]);
    }
#else
    VMLFN_CALL(sLn,HAynn,((int)n, in, out))
#endif
}

template<CpuType cpu>
void vLog(size_t n, double* in, double* out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = log(in[i]);
    }
#else
    VMLFN_CALL(dLn,HAynn,((int)n, in, out))
#endif
}

template<CpuType cpu>
inline float sLog(float in)
{
    float r;

    vLog<cpu>(1,&in,&r);

    return r;
}

template<CpuType cpu>
inline double sLog(double in)
{
    double r;

    vLog<cpu>(1,&in,&r);

    return r;
}


/**
*** vector and scalar Log1p functions (vLog1p / sLog1p)
**/

template<CpuType cpu>
void vLog1p(size_t n, float *in, float *out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = log1pf(in[i]);
    }
#else
    VMLFN_CALL(sLog1p,HAynn,((int)n, in, out))
#endif
}

template<CpuType cpu>
void vLog1p(size_t n, double* in, double* out)
{
#if defined(_DAAL_USE_SVML)
  #pragma ivdep
    for( size_t i=0; i<n; i++ )
    {
        out[i] = log1p(in[i]);
    }
#else
    VMLFN_CALL(dLog1p,HAynn,((int)n, in, out))
#endif
}


/**
*** vector and scalar CdfNormInv functions
**/

template<CpuType cpu>
void vCdfNormInv(size_t n, float *in, float *out)
{
    VMLFN_CALL(sCdfNormInv,HAynn,((int)n, in, out))
}

template<CpuType cpu>
void vCdfNormInv(size_t n, double* in, double* out)
{
    VMLFN_CALL(dCdfNormInv,HAynn,((int)n, in, out))
}

template<CpuType cpu>
inline float sCdfNormInv(float in)
{
    float r;

    vCdfNormInv<cpu>(1,&in,&r);

    return r;
}

template<CpuType cpu>
inline double sCdfNormInv(double in)
{
    double r;

    vCdfNormInv<cpu>(1,&in,&r);

    return r;
}

}

#endif
