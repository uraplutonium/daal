/* file: service_dnn.h */
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
//  Template wrappers for DNN functions.
//--
*/


#ifndef __SERVICE_DNN_H__
#define __SERVICE_DNN_H__

#include "daal_defines.h"
#include "mkl_dnn_types.h"

#include "service_dnn_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklDnn>
struct Dnn
{
    typedef typename _impl<fpType,cpu>::ErrorType ErrorType;
    typedef typename _impl<fpType,cpu>::PrimitiveType PrimitiveType;
    typedef typename _impl<fpType,cpu>::LayoutType LayoutType;
    typedef typename _impl<fpType,cpu>::AlgorithmType AlgorithmType;
    typedef typename _impl<fpType,cpu>::BorderType BorderType;
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static ErrorType xConvolutionCreateForwardBias(
        PrimitiveType* pConvolution, AlgorithmType algorithm, SizeType dimension,
        const SizeType srcSize[], const SizeType dstSize[], const SizeType filterSize[],
        const SizeType convolutionStrides[], const int inputOffset[],
        const BorderType border_type)
    {
        return _impl<fpType,cpu>::xConvolutionCreateForwardBias(pConvolution, algorithm, dimension, srcSize, dstSize,
                                                    filterSize, convolutionStrides, inputOffset, border_type);
    }
    static ErrorType xConvolutionCreateBackwardData(
        PrimitiveType* pConvolution, AlgorithmType algorithm, SizeType dimension,
        const SizeType srcSize[], const SizeType dstSize[], const SizeType filterSize[],
        const SizeType convolutionStrides[], const int inputOffset[],
        const BorderType border_type)
    {
        return _impl<fpType,cpu>::xConvolutionCreateBackwardData(pConvolution, algorithm, dimension, srcSize, dstSize, filterSize,
                                                     convolutionStrides, inputOffset, border_type);
    }
    static ErrorType xConvolutionCreateBackwardFilter(
        PrimitiveType* pConvolution, AlgorithmType algorithm, SizeType dimension,
        const SizeType srcSize[], const SizeType dstSize[], const SizeType filterSize[],
        const SizeType convolutionStrides[], const int inputOffset[],
        const BorderType border_type)
    {
        return _impl<fpType,cpu>::xConvolutionCreateBackwardFilter(pConvolution, algorithm, dimension, srcSize, dstSize, filterSize,
                                                       convolutionStrides, inputOffset, border_type);
    }
    static ErrorType xConvolutionCreateBackwardBias(
        PrimitiveType* pConvolution, AlgorithmType algorithm, SizeType dimension,
        const SizeType dstSize[])
    {
        return _impl<fpType,cpu>::xConvolutionCreateBackwardBias(pConvolution, algorithm, dimension, dstSize);
    }

    static ErrorType xExecute(PrimitiveType primitive, void *resources[])
    {
        return _impl<fpType,cpu>::xExecute(primitive, resources);
    }

    static ErrorType xConversionExecute(PrimitiveType conversion, void *from, void *to)
    {
        return _impl<fpType,cpu>::xConversionExecute(conversion, from, to);
    }

    static ErrorType xLayoutCreate(LayoutType *pLayout, SizeType dimension, const SizeType size[], const SizeType strides[])
    {
        return _impl<fpType,cpu>::xLayoutCreate(pLayout, dimension, size, strides);
    }

    static ErrorType xLayoutCreateFromPrimitive(LayoutType *pLayout, const PrimitiveType primitive, dnnResourceType_t type)
    {
        return _impl<fpType,cpu>::xLayoutCreateFromPrimitive(pLayout, primitive, type);
    }

    static ErrorType xAllocateBuffer(void **pPtr, LayoutType layout)
    {
        return _impl<fpType,cpu>::xAllocateBuffer(pPtr, layout);
    }

    static ErrorType xReleaseBuffer(void *ptr)
    {
        return _impl<fpType,cpu>::xReleaseBuffer(ptr);
    }

    static int xLayoutCompare(const LayoutType l1, const LayoutType l2)
    {
        return _impl<fpType,cpu>::xLayoutCompare(l1, l2);
    }

    static ErrorType xConversionCreate(PrimitiveType* pConversion, const LayoutType from, const LayoutType to)
    {
        return _impl<fpType,cpu>::xConversionCreate(pConversion, from, to);
    }

    static ErrorType xDelete(PrimitiveType primitive)
    {
        return _impl<fpType,cpu>::xDelete(primitive);
    }

    static ErrorType xLayoutDelete(LayoutType layout)
    {
        return _impl<fpType,cpu>::xLayoutDelete(layout);
    }

};

} // namespace internal
} // namespace daal

#endif
