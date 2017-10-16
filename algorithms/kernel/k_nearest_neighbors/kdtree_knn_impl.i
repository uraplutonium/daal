/* file: kdtree_knn_impl.i */
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
//  Common functions for K-Nearest Neighbors
//--
*/

#ifndef __KDTREE_KNN_IMPL_I__
#define __KDTREE_KNN_IMPL_I__

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif


#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    #define DAAL_PREFETCH_READ_T0(addr) _mm_prefetch((char *)addr, _MM_HINT_T0)
#else
    #define DAAL_PREFETCH_READ_T0(addr) __builtin_prefetch((char *)addr, 0, 3)
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900)
    #define DAAL_ALIGNAS(n) __declspec(align(n))
#else
    #define DAAL_ALIGNAS(n) alignas(n)
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace internal
{

#define __KDTREE_MAX_NODE_COUNT_MULTIPLICATION_FACTOR 3
#define __KDTREE_LEAF_BUCKET_SIZE 31 // Must be ((power of 2) minus 1).
#define __KDTREE_FIRST_PART_LEAF_NODES_PER_THREAD 3
#define __KDTREE_DIMENSION_SELECTION_SIZE 128
#define __KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT 1024
#define __KDTREE_DEPTH_MULTIPLICATION_FACTOR 4
#define __KDTREE_SEARCH_SKIP 32
#define __KDTREE_INDEX_VALUE_PAIRS_PER_THREAD 8192
#define __KDTREE_SAMPLES_PERCENT 0.5
#define __KDTREE_MAX_SAMPLES 1024
#define __KDTREE_MIN_SAMPLES 256
#define __KDTREE_THREAD_LOCAL_BUFFER 1024
#define __KDTREE_MAX_QUERIES_PER_ROUND 65536
#define __SIMDWIDTH 8

#define __KDTREE_NULLDIMENSION (static_cast<size_t>(-1))

template <CpuType cpu, typename T>
inline const T & min(const T & a, const T & b) { return !(b < a) ? a : b; }

template <CpuType cpu, typename T>
inline const T & max(const T & a, const T & b) { return (a < b) ? b : a; }

template <typename algorithmFpType, CpuType cpu>
int compareFp(const void * p1, const void * p2)
{
    const algorithmFpType & v1 = *static_cast<const algorithmFpType *>(p1);
    const algorithmFpType & v2 = *static_cast<const algorithmFpType *>(p2);
    return (v1 < v2) ? -1 : 1;
}

template <CpuType cpu, typename ForwardIterator, typename T>
ForwardIterator lowerBound(ForwardIterator first, ForwardIterator last, const T & value)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it = first;
        const auto step = count / 2;
        it += step; // advance.
        if (*it < value)
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <typename T, CpuType cpu>
class Stack
{
public:
    Stack() : _data(nullptr) {}

    ~Stack() { services::daal_free(_data); }

    bool init(size_t size)
    {
        _data = static_cast<T *>(services::daal_malloc(size * sizeof(T)));
        _size = size;
        _top = _sizeMinus1 = size - 1;
        _count = 0;
        return _data;
    }

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset()
    {
        _top = _sizeMinus1;
        _count = 0;
    }

    DAAL_FORCEINLINE void push(const T & value)
    {
        if (_count >= _size)
        {
            grow();
        }

        _top = (_top + 1) & _sizeMinus1;
        _data[_top] = value;
        ++_count;
    }

    DAAL_FORCEINLINE T pop()
    {
        const T value = _data[_top--];
        _top = _top & _sizeMinus1;
        --_count;
        return value;
    }

    bool empty() const { return (_count == 0); }

    size_t size() const { return _count; }

    void grow()
    {
        _size *= 2;
        T * const newData = static_cast<T *>(services::daal_malloc(_size * sizeof(T)));
        if (_top == _sizeMinus1)
        {
            _top = _size - 1;
        }
        _sizeMinus1 = _size - 1;
        services::daal_memcpy_s(newData, _size * sizeof(T), _data, _count * sizeof(T));
        T * const oldData = _data;
        _data = newData;
        services::daal_free(oldData);
    }

private:
    T * _data;
    size_t _top;
    size_t _count;
    size_t _size;
    size_t _sizeMinus1;
};


template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapLeftChildIndex(T index) { return 2 * index + 1; }
template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapRightChildIndex(T index) { return 2 * index + 2; }
template <CpuType cpu, typename T> DAAL_FORCEINLINE T heapParentIndex(T index) { return (index - 1) / 2; }

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    if (first != last)
    {
        --last;
        auto i = last - first;
        if (i > 0)
        {
            const auto newItem = *last; // It can be moved instead.
            auto prev = i;
            for (i = heapParentIndex<cpu>(i); i && compare(*(first + i), newItem); i = heapParentIndex<cpu>(i))
            {
                *(first + prev) = *(first + i); // It can be moved instead.
                prev = i;
            }
            *(first + i) = newItem; // It can be moved instead.
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    /*

        while (child && _elements[parnt] < newItem) {
            _elements[child] = _elements[parnt];
            child = parnt;
            parnt = heapParentIndex(child);
        }
        _elements[child] = newItem;
     */

    if (first != last)
    {
        --last;
        auto i = last - first;
        if (i > 0)
        {
            const auto newItem = *last;
            auto j = heapParentIndex<cpu>(i);
            while (i && (*(first + j) < newItem))
            {
                *(first + i) = *( first + j);
                i = j;
                j = heapParentIndex<cpu>(i);
            }
            *(first + i) = newItem;

            // const auto newItem = *last; // It can be moved instead.
            // for (i = heapParentIndex<cpu>(i); i && (*(first + i) < newItem); i = heapParentIndex<cpu>(i))
            // {
            //     *(first + prev) = *(first + i); // It can be moved instead.
            //     prev = i;
            // }
            // *(first + i) = newItem; // It can be moved instead.
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff, typename Compare>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i, Compare compare)
{
    for (auto largest = i;;)
    {
        const auto l = heapLeftChildIndex<cpu>(i);
        if ((l < count) && compare(*(first + largest), *(first + l)))
        {
            largest = l;
        }
        const auto r = heapRightChildIndex<cpu>(i);
        if ((r < count) && compare(*(first + largest), *(first + r)))
        {
            largest = r;
        }

        if (largest == i) { break; }
        auto temp = *(first + i); *(first + i) = *(first + largest); *(first + largest) = temp; // Moving can be used instead.
        i = largest;
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i)
{
    for (auto largest = i;; i = largest)
    {
        const auto l = heapLeftChildIndex<cpu>(i);
        if ((l < count) && (*(first + largest) < *(first + l)))
        {
            largest = l;
        }
        const auto r = heapRightChildIndex<cpu>(i);
        if ((r < count) && (*(first + largest) < *(first + r)))
        {
            largest = r;
        }

        if (largest == i) { break; }
        auto temp = *(first + i); *(first + i) = *(first + largest); *(first + largest) = temp; // Moving can be used instead.
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    if (1 < last - first)
    {
        --last;
        auto temp = *first; *first = *last; *last = temp; // Moving can be used instead.
        internalAdjustMaxHeap<cpu>(first, last, last - first, first - first, compare);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    if (1 < last - first)
    {
        --last;
        auto temp = *first; *first = *last; *last = temp; // Moving can be used instead.
        internalAdjustMaxHeap<cpu>(first, last, last - first, first - first);
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    const auto count = last - first;
    auto i = count / 2;
    while (0 < i) { internalAdjustMaxHeap<cpu>(first, last, count, --i, compare); }
}

template <CpuType cpu, typename RandomAccessIterator>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    const auto count = last - first;
    auto i = count / 2;
    while (0 < i) { internalAdjustMaxHeap<cpu>(first, last, count, --i); }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
RandomAccessIterator isMaxHeapUntil(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    if (first != last)
    {
        auto i = first;
        while (++i != last)
        {
            if (compare(*(first + heapParentIndex<cpu>(i)), *i)) { return i; }
        }
    }
    return last;
}

template <CpuType cpu, typename RandomAccessIterator>
RandomAccessIterator isMaxHeapUntil(RandomAccessIterator first, RandomAccessIterator last)
{
    if (first != last)
    {
        auto i = first;
        while (++i != last)
        {
            if (*(first + heapParentIndex<cpu>(i)) < *i) { return i; }
        }
    }
    return last;
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
DAAL_FORCEINLINE bool isMaxHeap(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    return (isMaxHeapUntil<cpu>(first, last, compare) == last);
}

template <CpuType cpu, typename RandomAccessIterator>
DAAL_FORCEINLINE bool isMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    return (isMaxHeapUntil<cpu>(first, last) == last);
}

template <typename T, CpuType cpu>
class Heap
{
public:
    Heap() : _elements(nullptr), _count(0) {}

    ~Heap() { services::daal_free(_elements); }

    bool init(size_t size)
    {
        DAAL_ASSERT(_elements == nullptr);
        _count = 0;
        _elements = static_cast<T *>(services::daal_malloc(size * sizeof(T)));
        return _elements;
    }

    void clear()
    {
        if (_elements)
        {
            services::daal_free(_elements);
            _elements = nullptr;
        }
    }

    void reset() { _count = 0; }

    void push(const T & e, size_t k)
    {
        _elements[_count++] = e;
        if (_count == k) { makeMaxHeap<cpu>(_elements, _elements + _count); }
    }

    void replaceMax(const T & e)
    {
        popMaxHeap<cpu>(_elements, _elements + _count);
        _elements[_count - 1] = e;
        pushMaxHeap<cpu>(_elements, _elements + _count);
    }

    void removeMax()
    {
        popMaxHeap<cpu>(_elements, _elements + _count);
        --_count;
    }

    size_t size() const { return _count; }

    T * getMax() { return _elements; }

    const T * getMax() const { return _elements; }

    const T & operator[] (size_t index) const { return *(_elements + index); }

private:
    T * _elements;
    size_t _count;
};

} // namespace internal
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
