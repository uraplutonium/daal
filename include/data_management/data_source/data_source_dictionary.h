/* file: data_source_dictionary.h */
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
//  Implementation of a data source dictionary.
//--
*/

#ifndef __DATA_SOURCE_DICTIONARY_H__
#define __DATA_SOURCE_DICTIONARY_H__

#include <string>
#include <map>
#include "data_management/data/data_dictionary.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
class CategoricalFeatureDictionary : public std::map<std::string, std::pair<int, int> >
{
public:
};

/**
 *  <a name="DAAL-CLASS-DATASOURCEFEATURE"></a>
 *  \brief Data structure that describes the Data Source feature
 */
class DataSourceFeature : public SerializationIface
{
public:
    NumericTableFeature             ntFeature;
    size_t                          name_length;
    char                           *name;

    CategoricalFeatureDictionary   *cat_dict;

public:
    /**
     *  Constructor of a data feature
     */
    DataSourceFeature() : name(NULL), name_length(0), cat_dict(NULL) {}

    /**
     *  Copy operator for a data feature
     */
    DataSourceFeature &operator= (const DataSourceFeature &f)
    {
        ntFeature = f.ntFeature;
        name                = f.name           ;
        name_length         = f.name_length    ;
        cat_dict            = 0;

        return *this;
    }

    /** \private */
    virtual ~DataSourceFeature()
    {
        if(name)
        {
            delete[] name;
        }

        if(cat_dict)
        {
            delete cat_dict;
        }
    }

    /**
     *  Gets a categorical features dictionary
     *  \return Pointer to the categorical features dictionary
     */
    CategoricalFeatureDictionary *getCategoricalDictionary()
    {
        if( !cat_dict )
        {
            cat_dict = new CategoricalFeatureDictionary;
        }

        return cat_dict;
    }

    /**
     *  Specifies the name of a data feature
     *  \param[in]  featureName  Name of the data feature
     */
    void setFeatureName(const std::string &featureName)
    {
        name_length = featureName.length() + 1;
        if (name)
        {
            delete[] name;
        }
        name = new char[name_length];
        daal::services::daal_memcpy_s(name, name_length, featureName.c_str(), name_length);
    }

    /**
     *  Fills the class based on a specified type
     *  \tparam  T  Name of the data feature
     */
    template<typename T>
    void setType()
    {
        ntFeature.setType<T>();
    }

    /** \private */
    void serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->setObj( &ntFeature );

        arch->set( name_length );

        if( onDeserialize )
        {
            if( name_length > 0 )
            {
                if( name ) { delete[] name; }
                name = NULL;
                name = new char[name_length];
            }
        }

        arch->set( name, name_length );
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_DATAFEATURE_NT_ID;
    }
};

typedef Dictionary<DataSourceFeature, SERIALIZATION_DATADICTIONARY_DS_ID>   DataSourceDictionary;

} // namespace interface1

using interface1::CategoricalFeatureDictionary;
using interface1::DataSourceFeature;
using interface1::DataSourceDictionary;

}
} // namespace daal
#endif
