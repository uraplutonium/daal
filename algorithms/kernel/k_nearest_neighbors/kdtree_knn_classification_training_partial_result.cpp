/* file: kdtree_knn_classification_training_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

/*
//++
//  Implementation of K-Nearest Neighbors (kNN) algorithm classes.
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep4, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep5, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP5_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep6, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP6_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep7, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP7_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep8, SERIALIZATION_K_NEAREST_NEIGHBOR_DISTRIBUTED_PARTIAL_RESULT_STEP8_ID);

DistributedPartialResultStep1::DistributedPartialResultStep1() : daal::algorithms::PartialResult(1) {}

NumericTablePtr DistributedPartialResultStep1::get(DistributedPartialResultStep1Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep1::set(DistributedPartialResultStep1Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep1::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    // const Input *algInput = static_cast<const InputIface *>(input);

    // size_t nUsers = algInput->getNumberOfUsers();
    // size_t nItems = algInput->getNumberOfItems();

    // int unexpectedLayouts = (int)packed_mask;
    // if(result)
    // {
    //     if(!checkNumericTable(result->get(prediction).get(), this->_errors.get(), predictionStr(), unexpectedLayouts, 0, nItems, nUsers)) { return; }
    // }
    return services::Status();
}

DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(2) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep2::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(3) {}

NumericTablePtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep3::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep4::DistributedPartialResultStep4() : daal::algorithms::PartialResult(1) {}

NumericTablePtr DistributedPartialResultStep4::get(DistributedPartialResultStep4Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep4::set(DistributedPartialResultStep4Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep4::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep5::DistributedPartialResultStep5() : daal::algorithms::PartialResult(4) {}

NumericTablePtr DistributedPartialResultStep5::get(DistributedPartialResultStep5Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep5::set(DistributedPartialResultStep5Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep5::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep6::DistributedPartialResultStep6() : daal::algorithms::PartialResult(2) {}

NumericTablePtr DistributedPartialResultStep6::get(DistributedPartialResultStep6Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep6::set(DistributedPartialResultStep6Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep6::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep7::DistributedPartialResultStep7() : daal::algorithms::PartialResult(2) {}

NumericTablePtr DistributedPartialResultStep7::get(DistributedPartialResultStep7Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

SharedPtr<kdtree_knn_classification::PartialModel> DistributedPartialResultStep7::get(DistributedPartialResultStep7PartialModelId id) const
{
    return services::staticPointerCast<kdtree_knn_classification::PartialModel, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep7::set(DistributedPartialResultStep7Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep7::set(DistributedPartialResultStep7PartialModelId id,
                                        const SharedPtr<kdtree_knn_classification::PartialModel> & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep7::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

DistributedPartialResultStep8::DistributedPartialResultStep8() : daal::algorithms::PartialResult(1) {}

SharedPtr<kdtree_knn_classification::PartialModel> DistributedPartialResultStep8::get(DistributedPartialResultStep8Id id) const
{
    return services::staticPointerCast<kdtree_knn_classification::PartialModel, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep8::set(DistributedPartialResultStep8Id id, const SharedPtr<kdtree_knn_classification::PartialModel> & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep8::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
