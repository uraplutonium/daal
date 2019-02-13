/* file: PartialSVDResult.java */
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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALSVDRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() of the %SVD method of the PCA algorithm
 *        in the online or distributed processing mode
 */
public final class PartialSVDResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result of the algorithm
     * @param context       Context to manage the partial result of the algorithm
     */
    public PartialSVDResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public PartialSVDResult(DaalContext context, long cPartialResult) {
        super(context, cPartialResult);
    }

    /**
     * Returns partial result of the PCA algorithm
     * @param  id   Identifier of the PartialSVDResult, @ref PartialSVDTableResultID
     * @return      Partial result that corresponds to the given identifier
     */
    public NumericTable get(PartialSVDTableResultID id) {
        int idValue = id.getValue();
        if (id != PartialSVDTableResultID.nObservations && id != PartialSVDTableResultID.sumSVD
                && id != PartialSVDTableResultID.sumSquaresSVD) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialResultValue(getCObject(), idValue));
    }

    /**
     * Sets partial result of the PCA algorithm
     * @param id                 Identifier of the partial result
     * @param value              Value of the partial result
     */
    public void set(PartialSVDTableResultID id, NumericTable value) {
        int idValue = id.getValue();
        if (id != PartialSVDTableResultID.nObservations && id != PartialSVDTableResultID.sumSVD
                && id != PartialSVDTableResultID.sumSquaresSVD) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultValue(getCObject(), idValue, value.getCObject());
    }

    /**
     * Returns partial result of the PCA algorithm
     * @param  id   Identifier of the PartialSVDResult, @ref PartialSVDTableResultID
     * @return Partial result that corresponds to the given identifier
     */
    public DataCollection get(PartialSVDCollectionResultID id) {
        int idValue = id.getValue();
        if (id != PartialSVDCollectionResultID.svdAuxiliaryData) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetPartialResultCollection(getCObject(), idValue));
    }

    /**
     * Sets partial result of the PCA algorithm
     * @param id                 Identifier of the partial result
     * @param value              Value of the partial result
     */
    public void set(PartialSVDCollectionResultID id, DataCollection value) {
        int idValue = id.getValue();
        if (id != PartialSVDCollectionResultID.svdAuxiliaryData) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultCollection(getCObject(), idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResultValue(long cPartialResult, int id);

    private native void cSetPartialResultValue(long cPartialResult, int id, long cObject);

    private native long cGetPartialResultCollection(long cPartialResult, int id);

    private native void cSetPartialResultCollection(long cPartialResult, int id, long cObject);
}
/** @} */
