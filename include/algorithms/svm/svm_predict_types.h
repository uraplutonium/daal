/* file: svm_predict_types.h */
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
//  SVM parameter structure
//--
*/

#ifndef __SVM_PREDICT_TYPES_H__
#define __SVM_PREDICT_TYPES_H__

#include "algorithms/classifier/classifier_predict_types.h"
#include "algorithms/svm/svm_model.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
/**
 * \brief Contains classes to make predictions based on the SVM model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVM__PREDICTION__METHOD"></a>
 * Available methods to run predictions based on the SVM model
 */
enum Method
{
    defaultDense = 0          /*!< Default SVM model-based prediction method */
};

} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
