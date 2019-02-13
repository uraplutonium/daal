/* file: InitDistributedStep3MasterPlusPlusPartialResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP3MASTERPLUSPLUSPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing initial centroids for
 *        the K-Means algorithm in the distributed processing mode
 */
public final class InitDistributedStep3MasterPlusPlusPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs empty object
     * @param context       Context to manage the partial result of computing initial centroids for the K-Means algorithm
     */
    public InitDistributedStep3MasterPlusPlusPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitDistributedStep3MasterPlusPlusPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of computing initial centroids for the K-Means algorithm
     * @param id  Identifier of the partial result object
     * @param key Identifier of the node the object comes from
     * @return    Partial result object that corresponds to the given identifier
     */
    public NumericTable get(InitDistributedStep3MasterPlusPlusPartialResultId id, int key) {
        if (id == InitDistributedStep3MasterPlusPlusPartialResultId.outputOfStep3ForStep4) {
            long tbl = cGetTable(cObject, id.getValue(), key);
            if(tbl == 0)
                return null;
            return (NumericTable)Factory.instance().createObject(getContext(), tbl);
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an partial result for computing initial centroids for the K-Means algorithm
     * in the 3rd step in the distributed processing mode
     * @param id    Identifier of the input object
     */

    public SerializableBase get(InitDistributedStep3MasterPlusPlusPartialResultDataId id) {
        if (id != InitDistributedStep3MasterPlusPlusPartialResultDataId.outputOfStep3ForStep5) {
            throw new IllegalArgumentException("id unsupported");
        }
        long addr = cGetSerializableBase(cObject, id.getValue());
        if(addr == 0)
            return null;
        return Factory.instance().createObject(getContext(), addr);
    }

    /**
    * Returns a partial result object for computing initial centroids for the K-Means algorithm
    * @param id Identifier of the partial result object
    * @return   Partial result object that corresponds to the given identifier
    */
    public KeyValueDataCollection get(InitDistributedStep3MasterPlusPlusPartialResultId id) {
        if (id != InitDistributedStep3MasterPlusPlusPartialResultId.outputOfStep3ForStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new KeyValueDataCollection(getContext(), cGetKeyValueDataCollection(cObject, id.getValue()));
    }

    private native long cNewPartialResult();

    private native long cGetTable(long inputAddr, int id, int key);

    private native long cGetKeyValueDataCollection(long inputAddr, int id);

    private native long cGetSerializableBase(long inputAddr, int id);
}
/** @} */
