/** file merged_numeric_table.cpp */
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

#include "merged_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

MergedNumericTable::MergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

MergedNumericTable::MergedNumericTable(NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(table);
}

MergedNumericTable::MergedNumericTable(NumericTablePtr first, NumericTablePtr second) :
    NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(first);
    this->_status |= addNumericTable(second);
}

MergedNumericTable::MergedNumericTable(services::Status &st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables) { st.add(services::ErrorMemoryAllocationFailed); }
    this->_status |= st;
}

MergedNumericTable::MergedNumericTable(const NumericTablePtr &table, services::Status &st) :
    NumericTable(0, 0),
    _tables(new DataCollection)
{
    if (!_tables) { st.add(services::ErrorMemoryAllocationFailed); }
    st |= addNumericTable(table);
    this->_status |= st;
}

MergedNumericTable::MergedNumericTable(const NumericTablePtr &first, const NumericTablePtr &second, services::Status &st) :
    NumericTable(0, 0),
    _tables(new DataCollection)
{
    if (!_tables) { st.add(services::ErrorMemoryAllocationFailed); }
    st |= addNumericTable(first);
    st |= addNumericTable(second);
    this->_status |= st;
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL(MergedNumericTable);
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(const NumericTablePtr &nestedTable, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(MergedNumericTable, nestedTable);
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(const NumericTablePtr &first,
                                                                   const NumericTablePtr &second,
                                                                   services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(MergedNumericTable, first, second);
}

}
}
}
