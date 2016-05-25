/* file: service_numeric_table.h */
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
//  CPU-specified homogeneous numeric table
//--
*/

#ifndef __SERVICE_NUMERIC_TABLE_H__
#define __SERVICE_NUMERIC_TABLE_H__

#include "homogen_numeric_table.h"
#include "service_defines.h"

using namespace daal::data_management;

namespace daal
{
namespace internal
{

template <CpuType cpu>
class NumericTableFeatureCPU : public NumericTableFeature
{
public:
    NumericTableFeatureCPU() : NumericTableFeature() {}
    virtual ~NumericTableFeatureCPU() {}
};

template <CpuType cpu>
class NumericTableDictionaryCPU : public NumericTableDictionary
{
public:
    NumericTableDictionaryCPU( size_t nfeat )
    {
        _nfeat = 0;
        _dict  = 0;
        if(nfeat) { setNumberOfFeatures(nfeat); }
    };

    void setAllFeatures(const NumericTableFeature &defaultFeature) DAAL_C11_OVERRIDE
    {
        for( size_t i = 0 ; i < _nfeat ; i++ )
        {
            _dict[i] = defaultFeature;
        }
    }

    void setNumberOfFeatures(size_t nfeat) DAAL_C11_OVERRIDE
    {
        if( _nfeat != 0 )
        {
            this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;
        }
        _nfeat = nfeat;
        _dict  = (NumericTableFeature *)(new NumericTableFeatureCPU<cpu>[_nfeat]);
    }
};

template <typename T, CpuType cpu>
class HomogenNumericTableCPU {};

template <CpuType cpu>
class HomogenNumericTableCPU<float, cpu> : public HomogenNumericTable<float>
{
public:
    HomogenNumericTableCPU( float *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<float>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<float>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        setNumberOfRows( obsnum );
        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template <CpuType cpu>
class HomogenNumericTableCPU<double, cpu> : public HomogenNumericTable<double>
{
public:
    HomogenNumericTableCPU( double *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<double>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<double>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        setNumberOfRows( obsnum );
        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template <CpuType cpu>
class HomogenNumericTableCPU<int, cpu> : public HomogenNumericTable<int>
{
public:
    HomogenNumericTableCPU( int *const ptr, size_t featnum, size_t obsnum )
        : HomogenNumericTable<int>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        _ptr = ptr;
        _memStatus = userAllocated;
        setNumberOfRows( obsnum );
    }

    HomogenNumericTableCPU( size_t featnum, size_t obsnum )
        : HomogenNumericTable<int>(new NumericTableDictionaryCPU<cpu>(featnum))
    {
        _cpuDict = _ddict.get();
        setNumberOfRows( obsnum );
        allocateDataMemory();
    }

    virtual ~HomogenNumericTableCPU() {
        delete _cpuDict;
    }

    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

private:
    NumericTableDictionary *_cpuDict;
};

template<typename algorithmFPType, CpuType cpu, typename NumericTableType>
class ReadRows
{
public:
    ReadRows(const NumericTableType& data, size_t nRows): m_data(data)
    {
        const_cast<NumericTableType&>(m_data).getBlockOfRows(0, nRows, readOnly, m_block);
    }
    const algorithmFPType* get() const { return m_block.getBlockPtr(); }
    ~ReadRows() { const_cast<NumericTableType&>(m_data).releaseBlockOfRows(m_block); }

private:
    const NumericTableType& m_data;
    mutable BlockDescriptor<algorithmFPType> m_block;
};

template<typename algorithmFPType, CpuType cpu, typename NumericTableType>
class WriteRows
{
public:
    WriteRows(NumericTableType& data, size_t nRows): m_data(data)
    {
        m_data.getBlockOfRows(0, nRows, readWrite, m_block);
    }
    algorithmFPType* get() { return m_block.getBlockPtr(); }
    ~WriteRows() { m_data.releaseBlockOfRows(m_block); }

private:
    NumericTableType& m_data;
    BlockDescriptor<algorithmFPType> m_block;
};

template<CpuType cpu>
class SmartPtr
{
public:
    SmartPtr(size_t n): m_data(nullptr) { m_data = daal::services::daal_malloc(n); }
    ~SmartPtr() { if(m_data) daal::services::daal_free(m_data); }
    void* get() { return m_data; }

private:
    void* m_data;
};

} // internal namespace
} // daal namespace

#endif
