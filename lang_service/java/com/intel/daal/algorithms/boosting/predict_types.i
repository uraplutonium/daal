/* file: predict_types.i */
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
#include "boosting/prediction/JNumericTableInputId.h"
#include "boosting/prediction/JPredictionResultId.h"
#include "boosting/prediction/JModelInputId.h"

using namespace daal;
using namespace daal::algorithms;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define Data com_intel_daal_algorithms_boosting_prediction_NumericTableInputId_Data

#define ModelInput com_intel_daal_algorithms_boosting_prediction_ModelInputId_Model

#define PredictionResults com_intel_daal_algorithms_boosting_prediction_PredictionResultId_PredictionResults

typedef boosting::prediction::Batch b_pr_of;
