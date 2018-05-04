/* file: common_helpers_argument.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
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

#include "daal.h"

namespace daal
{

using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;

template<typename _Input>
struct jniInput
{
    template<typename _IdType, typename _DataType, typename... Args>
    static jlong get( jlong inputAddr, jint id, Args&&... args)
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
        *dShPtr = staticPointerCast<SerializationIface, _DataType>( input->get( (_IdType)id, args... ) );
        return (jlong)dShPtr;
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static void set( jlong inputAddr, jint id, jlong dataAddr, Args&&... args )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->set((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr), args...);
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong inputAddr, jint id, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->add((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong inputAddr, jint id, jint key, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->add((_IdType)id, (size_t)key, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _WidType, typename _IdType, typename _DataType>
    static void setex( jlong inputAddr, jint wid, jint id, jlong dataAddr )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        input->set((_WidType)wid, (_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }

    template<typename _WidType, typename _IdType, typename _DataType>
    static jlong getex( jlong inputAddr, jint wid, jint id )
    {
        _Input *input = (_Input*)inputAddr;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
        *dShPtr = staticPointerCast<SerializationIface, _DataType>( input->get( (_WidType)wid, (_IdType)id ) );
        return (jlong)dShPtr;
    }
};

template<typename _Argument>
struct jniArgument
{
    static jlong newObj()
    {
        return (jlong)( new SerializationIfacePtr( new _Argument() ) );
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static jlong get( jlong argumentAddr, jint id, Args&&... args )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        services::SharedPtr<_DataType> tmp = argument->get((_IdType)id, args...);
        if(!tmp)
            return (jlong)0;
        SerializationIfacePtr * dShPtr = new SerializationIfacePtr;
        *dShPtr = staticPointerCast<SerializationIface, _DataType>(tmp);
        return (jlong)dShPtr;
    }

    template<typename _IdType, typename _DataType, typename... Args>
    static void set( jlong argumentAddr, jint id, jlong dataAddr, Args&&... args )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        argument->set((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr), args...);
    }

    template<typename _IdType, typename _DataType>
    static void add( jlong argumentAddr, jint id, jlong dataAddr )
    {
        SharedPtr<_Argument> argument = staticPointerCast<_Argument, SerializationIface>( *(SerializationIfacePtr *)argumentAddr );
        SerializationIfacePtr *dataShPtr = (SerializationIfacePtr *)dataAddr;
        argument->add((_IdType)id, staticPointerCast<_DataType, SerializationIface>(*dataShPtr));
    }
};

}
