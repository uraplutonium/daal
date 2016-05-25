/* file: daal_strings.h */
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
//  String variables.
//--
*/

#ifndef __DAAL_STRINGS_H__
#define __DAAL_STRINGS_H__

/** \file daal_strings.h */

#include "services/daal_defines.h"

#define DAAL_STRINGS_LIST()\
    DECLARE_DAAL_STRINGS(strBasicStatisticsSum,      "basicStatisticsSum"     ) \
    DECLARE_DAAL_STRINGS(strSortedData,              "sortedData"             ) \
    DECLARE_DAAL_STRINGS(strNormalizedData,          "normalizedData"         ) \
    DECLARE_DAAL_STRINGS(strInputGradient,           "inputGradient"          ) \
    DECLARE_DAAL_STRINGS(strGradient,                "gradient"               ) \
    DECLARE_DAAL_STRINGS(strValue,                   "value"                  ) \
    DECLARE_DAAL_STRINGS(strData,                    "data"                   ) \
    DECLARE_DAAL_STRINGS(strWeights,                 "weights"                ) \
    DECLARE_DAAL_STRINGS(strBiases,                  "biases"                 ) \
    DECLARE_DAAL_STRINGS(strPopulationMean,          "populationMean"         ) \
    DECLARE_DAAL_STRINGS(strPopulationVariance,      "populationVariance"     ) \
    DECLARE_DAAL_STRINGS(strResultLayerData,         "resultLayerData"        ) \
    DECLARE_DAAL_STRINGS(strAuxData,                 "auxData"                ) \
    DECLARE_DAAL_STRINGS(strAuxWeights,              "auxWeights"             ) \
    DECLARE_DAAL_STRINGS(strAuxMean,                 "auxMean"                ) \
    DECLARE_DAAL_STRINGS(strAuxStandardDeviation,    "auxStandardDeviation"   ) \
    DECLARE_DAAL_STRINGS(strAuxPopulationMean,       "auxPopulationMean"      ) \
    DECLARE_DAAL_STRINGS(strAuxPopulationVariance,   "auxPopulationVariance"  ) \
    DECLARE_DAAL_STRINGS(strAuxProbabilities,        "auxProbabilities"       ) \
    DECLARE_DAAL_STRINGS(strAuxGroundTruth,          "auxGroundTruth"         ) \
    DECLARE_DAAL_STRINGS(strInputLayerData,          "inputLayerData"         ) \
    DECLARE_DAAL_STRINGS(strAuxMask,                 "auxMask"                ) \
    DECLARE_DAAL_STRINGS(strInputGradientCollection, "inputGradientCollection") \
    DECLARE_DAAL_STRINGS(strValueCollection,         "valueCollection"        ) \
    DECLARE_DAAL_STRINGS(strWeightDerivatives,       "weightDerivatives"      ) \
    DECLARE_DAAL_STRINGS(strBiasDerivatives,         "biasDerivatives"        ) \
    DECLARE_DAAL_STRINGS(strCorrelationDistance,     "correlationDistance"    ) \
    DECLARE_DAAL_STRINGS(strCosineDistance,          "cosineDistance"         ) \
    DECLARE_DAAL_STRINGS(strQuantiles,               "quantiles"              ) \
    DECLARE_DAAL_STRINGS(strQuantileOrders,          "quantileOrders"         ) \
    DECLARE_DAAL_STRINGS(strCovariance,              "covariance"             ) \
    DECLARE_DAAL_STRINGS(strCorrelation,             "correlation"            ) \
    DECLARE_DAAL_STRINGS(strMean,                    "mean"                   ) \
    DECLARE_DAAL_STRINGS(strSum,                     "sum"                    ) \
    DECLARE_DAAL_STRINGS(strPermutedColumns,         "permutedColumns"        ) \
    DECLARE_DAAL_STRINGS(strMatrixQ,                 "matrixQ"                ) \
    DECLARE_DAAL_STRINGS(strMatrixR,                 "matrixR"                ) \
    DECLARE_DAAL_STRINGS(strPermutationMatrix,       "permutationMatrix"      )

/**
 *  Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) namespace
 */
namespace daal
{

#define DECLARE_DAAL_STRINGS(arg1, arg2) DAAL_EXPORT const char *arg1();
    DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRINGS

};

#endif
