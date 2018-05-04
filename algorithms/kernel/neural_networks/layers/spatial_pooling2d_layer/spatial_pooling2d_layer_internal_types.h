/* file: spatial_pooling2d_layer_internal_types.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Common classes for 2D pooling layers
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_INTERNAL_PARAMETER_H__
#define __SPATIAL_POOLING2D_LAYER_INTERNAL_PARAMETER_H__

#include "service_utils.h"
#include "tensor.h"
#include "collection.h"
#include "service_blas.h"
#include "neural_networks/layers/pooling2d/pooling2d_layer_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace internal
{

enum Method
{
    maximum = 0,
    average = 1,
    stochastic = 2
};

template<CpuType cpu>
class CommonSpatialPoolingFunctions
{
public:
    static void setParameter(const pooling2d::Parameter &src, pooling2d::Parameter &dst)
    {
        dst.indices = src.indices;
        dst.kernelSizes = src.kernelSizes;
        dst.strides = src.strides;
        dst.paddings = src.paddings;
    }
};

}
}
}
}
}
}

#endif
