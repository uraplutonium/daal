/* file: kdtree_knn_classification_training_distributed_input.cpp */
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
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based training
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "service_defines.h"
#include "daal_strings.h"

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

DistributedInput<step2Master>::DistributedInput() : daal::algorithms::Input(1)
{
    Argument::set(inputOfStep2, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

void DistributedInput<step2Master>::set(DistributedInputStep2Id id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

KeyValueDataCollectionPtr DistributedInput<step2Master>::get(DistributedInputStep2Id id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

void DistributedInput<step2Master>::add(DistributedInputStep2Id id, size_t key, const NumericTablePtr & value)
{
    const KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return services::Status();
}

size_t DistributedInput<step2Master>::getNumberOfFeatures() const
{
    const KeyValueDataCollectionConstPtr collection = get(inputOfStep2);
    if (collection && (collection->size() != 0))
    {
        const NumericTable * const nt = static_cast<const NumericTable *>(collection->getValueByIndex(0).get());
        if (nt)
        {
            return nt->getNumberOfColumns();
        }
    }

    return 0;
}

DistributedInput<step3Local>::DistributedInput() : daal::algorithms::Input(8) {}

NumericTablePtr DistributedInput<step3Local>::get(DistributedInputStep3Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

int DistributedInput<step3Local>::get(DistributedInputStep3IntId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id))->getValue<int>(0, 0);
}

void DistributedInput<step3Local>::set(DistributedInputStep3Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

void DistributedInput<step3Local>::set(DistributedInputStep3IntId id, int value)
{
    typedef HomogenNumericTable<int> HNT;
    SharedPtr<HNT> table(new HNT(1, 1, NumericTable::doAllocate, value));
    Argument::set(id, table);
}

services::Status DistributedInput<step3Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step3Local>::getNumberOfFeatures() const
{
    const NumericTableConstPtr dataTable = get(dataForStep3);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step3Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;

    if (parameter)
    {
        DAAL_ASSERT(dynamic_cast<const Parameter *>(parameter));
        const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
        if (algParameter->nClasses < 2)
        {
            return s.add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
        }
    }

    const NumericTablePtr dataTable = get(dataForStep3);
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const size_t rowCount = dataTable->getNumberOfRows();
    const size_t columnCount = dataTable->getNumberOfColumns();

    const NumericTablePtr labelsTable = get(labelsForStep3);
    DAAL_CHECK_STATUS(s, checkNumericTable(labelsTable.get(), labelsStr(), 0, 0, 1, rowCount));

    const NumericTablePtr boundingBoxesTable = get(boundingBoxesForStep3);
    DAAL_CHECK_STATUS(s, checkNumericTable(boundingBoxesTable.get(), boundingBoxesStr(), 0, 0, columnCount, 0));

    return s;
}

DistributedInput<step4Local>::DistributedInput() : daal::algorithms::Input(6)
{
    Argument::set(samplesForStep4, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

void DistributedInput<step4Local>::set(DistributedInputStep4PerNodeId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step4Local>::set(DistributedInputStep4Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

KeyValueDataCollectionPtr DistributedInput<step4Local>::get(DistributedInputStep4PerNodeId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr DistributedInput<step4Local>::get(DistributedInputStep4Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedInput<step4Local>::add(DistributedInputStep4PerNodeId id, size_t key, const NumericTablePtr & value)
{
    const KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

services::Status DistributedInput<step4Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step4Local>::getNumberOfFeatures() const
{
    const NumericTableConstPtr dataTable = get(dataForStep4);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step4Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    return services::Status();
}

DistributedInput<step5Local>::DistributedInput() : daal::algorithms::Input(6)
{
    Argument::set(histogramForStep5, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

void DistributedInput<step5Local>::set(DistributedInputStep5PerNodeId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step5Local>::set(DistributedInputStep5Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

void DistributedInput<step5Local>::set(DistributedInputStep5BoolId id, bool value)
{
    typedef HomogenNumericTable<int> HNT;
    SharedPtr<HNT> table(new HNT(1, 1, NumericTable::doAllocate, value ? 1 : 0));
    Argument::set(id, table);
}

KeyValueDataCollectionPtr DistributedInput<step5Local>::get(DistributedInputStep5PerNodeId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr DistributedInput<step5Local>::get(DistributedInputStep5Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

bool DistributedInput<step5Local>::get(DistributedInputStep5BoolId id) const
{
    return (services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id))->getValue<int>(0, 0) != 0);
}

void DistributedInput<step5Local>::add(DistributedInputStep5PerNodeId id, size_t key, const NumericTablePtr & value)
{
    const KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

services::Status DistributedInput<step5Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step5Local>::getNumberOfFeatures() const
{
    const NumericTableConstPtr dataTable = get(dataForStep5);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step5Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    return services::Status();
}

DistributedInput<step6Local>::DistributedInput() : daal::algorithms::Input(6) {}

void DistributedInput<step6Local>::set(DistributedInputStep6Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

NumericTablePtr DistributedInput<step6Local>::get(DistributedInputStep6Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

services::Status DistributedInput<step6Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step6Local>::getNumberOfFeatures() const
{
    const NumericTableConstPtr dataTable = get(dataForStep6);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step6Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    return services::Status();
}

DistributedInput<step7Master>::DistributedInput() : daal::algorithms::Input(6)
{
    Argument::set(dimensionForStep7, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    Argument::set(medianForStep7, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

void DistributedInput<step7Master>::set(DistributedInputStep7PerNodeId id, const KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step7Master>::set(DistributedInputStep7Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

void DistributedInput<step7Master>::set(DistributedInputStep7IntId id, int value)
{
    typedef HomogenNumericTable<int> HNT;
    SharedPtr<HNT> table(new HNT(1, 1, NumericTable::doAllocate, value));
    Argument::set(id, table);
}

void DistributedInput<step7Master>::set(DistributedInputStep7PartialModelId id, const SharedPtr<kdtree_knn_classification::PartialModel> & value)
{
    Argument::set(id, value);
}

KeyValueDataCollectionPtr DistributedInput<step7Master>::get(DistributedInputStep7PerNodeId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr DistributedInput<step7Master>::get(DistributedInputStep7Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

int DistributedInput<step7Master>::get(DistributedInputStep7IntId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id))->getValue<int>(0, 0);
}

SharedPtr<kdtree_knn_classification::PartialModel> DistributedInput<step7Master>::get(DistributedInputStep7PartialModelId id) const
{
    return staticPointerCast<kdtree_knn_classification::PartialModel, SerializationIface>(Argument::get(id));
}

void DistributedInput<step7Master>::add(DistributedInputStep7PerNodeId id, size_t key, const NumericTablePtr & value)
{
    const KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

services::Status DistributedInput<step7Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step7Master>::getNumberOfFeatures() const
{
    const NumericTableConstPtr boundingBoxes = get(boundingBoxesForStep7);
    return boundingBoxes ? boundingBoxes->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step7Master>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    return services::Status();
}

DistributedInput<step8Local>::DistributedInput() : daal::algorithms::Input(4) {}

void DistributedInput<step8Local>::set(DistributedInputStep8Id id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

void DistributedInput<step8Local>::set(DistributedInputStep8PartialModelId id, const SharedPtr<kdtree_knn_classification::PartialModel> & value)
{
    Argument::set(id, value);
}

NumericTablePtr DistributedInput<step8Local>::get(DistributedInputStep8Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

SharedPtr<kdtree_knn_classification::PartialModel> DistributedInput<step8Local>::get(DistributedInputStep8PartialModelId id) const
{
    return staticPointerCast<kdtree_knn_classification::PartialModel, SerializationIface>(Argument::get(id));
}

services::Status DistributedInput<step8Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

size_t DistributedInput<step8Local>::getNumberOfFeatures() const
{
    const NumericTableConstPtr dataTable = get(dataForStep8);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step8Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
