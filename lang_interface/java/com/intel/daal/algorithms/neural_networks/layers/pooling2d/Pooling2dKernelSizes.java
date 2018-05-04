/* file: Pooling2dKernelSizes.java */
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

/**
 * @ingroup pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DKERNELSIZES"></a>
 * \brief Data structure representing the size of the 2D subtensor from which the element is computed
 */
public final class Pooling2dKernelSizes {
    private long[] size; /*!< Array of sizes of the 2D subtensor from which the element is computed */

    /**
    * Constructs KernelSizes with parameters
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public Pooling2dKernelSizes(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Sets the the array of sizes of the 2D subtensor from which the element is computed
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of sizes of the 2D subtensor from which the element is computed
    * @return Array of sizes of the 2D subtensor from which the element is computed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
