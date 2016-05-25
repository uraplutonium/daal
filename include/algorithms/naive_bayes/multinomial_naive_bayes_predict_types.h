/* file: multinomial_naive_bayes_predict_types.h */
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
//  Naive Bayes classifier parameter structure used in the prediction stage
//--
*/

#ifndef __NAIVE_BAYES_PREDICT_TYPES_H__
#define __NAIVE_BAYES_PREDICT_TYPES_H__

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "data_management/data/data_collection.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
/**
 * \brief Contains classes for multinomial naive Bayes model based prediction
 */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__METHOD"></a>
 * Available methods for computing the results of the naive Bayes model based prediction
 */
enum Method
{
    defaultDense = 0, /*!< Default multinomial naive Bayes model based prediction  */
    fastCSR      = 1  /*!< Multinomial naive Bayes model based prediction for sparse data in CSR format */
};

} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
