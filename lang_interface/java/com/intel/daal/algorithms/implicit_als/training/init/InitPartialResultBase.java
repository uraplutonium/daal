/* file: InitPartialResultBase.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULTBASE"></a>
 * @brief Provides interface to access partial results obtained with the implicit ALS initialization algorithm
 * in the first and second steps of the distributed processing mode
 */
public class InitPartialResultBase extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the partial result for the implicit ALS initialization algorithm in the distributed processing mode
     * @param context Context to manage the partial result for the implicit ALS initialization algorithm in the distributed processing mode
     */
    public InitPartialResultBase(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitPartialResultBase(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets a partial result of the implicit ALS initialization algorithm
     * @param id    Identifier of the partial result
     * @return Partial result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(InitPartialResultBaseId id) {
        if (id != InitPartialResultBaseId.outputOfInitForComputeStep3 &&
            id != InitPartialResultBaseId.offsets) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new KeyValueDataCollection(getContext(), cGetPartialResultCollection(cObject, id.getValue()));
    }

    /**
     * Gets a numeric table object from a partial result of the implicit ALS initialization algorithm
     * @param id    Identifier of the partial result
     * @param key   Key to use to retrieve a numeric table
     * @return Numeric table from the partial result that corresponds to the given identifier
     */
    public NumericTable get(InitPartialResultBaseId id, long key) {
        if (id != InitPartialResultBaseId.outputOfInitForComputeStep3 &&
            id != InitPartialResultBaseId.offsets) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialResultTable(cObject, id.getValue(), key));
    }

    /**
     * Sets the partial result of the implicit ALS initialization algorithm
     * @param id    Identifier of the partial result
     * @param value The partial result object
     */
    public void set(InitPartialResultBaseId id, KeyValueDataCollection value) {
        if (id != InitPartialResultBaseId.outputOfInitForComputeStep3 &&
            id != InitPartialResultBaseId.offsets) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultCollection(cObject, id.getValue(), value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResultCollection(long cResult, int id);
    private native void cSetPartialResultCollection(long cResult, int id, long cModel);

    private native long cGetPartialResultTable(long cResult, int id, long key);
}
/** @} */
