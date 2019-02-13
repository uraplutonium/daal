/* file: multiclassclassifier_train_oneagainstone_batch_fpt_cpu.cpp */
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

/*
//++
//  Implementation of One-Against-One method for Multi-class classifier
//  training algorithm.
//--
*/

#include "multiclassclassifier_train_batch_container.h"
#include "multiclassclassifier_train_kernel.h"
#include "multiclassclassifier_train_oneagainstone_kernel.h"
#include "multiclassclassifier_train_oneagainstone_impl.i"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, oneAgainstOne,    DAAL_CPU>;
}
namespace internal
{

template class MultiClassClassifierTrainKernel<oneAgainstOne,    DAAL_FPTYPE, DAAL_CPU>;

} // namespace internal

} // namespace training

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal
