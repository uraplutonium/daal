/* file: train_types.i */
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

#include "daal.h"

#include "JComputeMode.h"
#include "stump/training/JTrainingMethod.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_offlineValue
#define jOnline  com_intel_daal_algorithms_ComputeMode_onlineValue
#define jLocal   com_intel_daal_algorithms_ComputeMode_distributedLocalValue
#define jMaster  com_intel_daal_algorithms_ComputeMode_distributedMasterValue

#define DefaultDense com_intel_daal_algorithms_stump_training_TrainingMethod_DefaultDense

typedef stump::training::Batch<float, stump::training::defaultDense>     st_tr_of_s_dd;
typedef stump::training::Batch<double, stump::training::defaultDense>    st_tr_of_d_dd;
