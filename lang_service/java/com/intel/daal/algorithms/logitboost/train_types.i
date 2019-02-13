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
#include "logitboost/training/JTrainingMethod.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

#define jBatch   com_intel_daal_algorithms_ComputeMode_batchValue

#define Friedman com_intel_daal_algorithms_logitboost_training_TrainingMethod_Friedman

typedef logitboost::training::Batch<float, logitboost::training::friedman>     lb_tr_of_s_dd;
typedef logitboost::training::Batch<double, logitboost::training::friedman>    lb_tr_of_d_dd;

typedef SharedPtr<logitboost::training::Batch<float, logitboost::training::friedman> >    sp_lb_tr_of_s_dd;
typedef SharedPtr<logitboost::training::Batch<double, logitboost::training::friedman> >   sp_lb_tr_of_d_dd;
