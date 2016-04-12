/* file: data_dictionary.h */
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
//  Implementation of a data dictionary.
//--
*/

#ifndef __DATA_DICTIONARY_H__
#define __DATA_DICTIONARY_H__

#include <string>
#include <map>
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/data_utils.h"

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
 *  <a name="DAAL-CLASS-NUMERICTABLEDATAFEATURE"></a>
 *  \brief Data structure describes the Numeric Table feature
 */
class NumericTableFeature : public SerializationIface
{
public:
    data_feature_utils::IndexNumType      indexType;
    data_feature_utils::PMMLNumType       pmmlType;
    data_feature_utils::FeatureType       featureType;
    size_t                              typeSize;
    size_t                              categoryNumber;

public:
    /**
     *  Constructor of a data feature
     */
    NumericTableFeature()
    {
        indexType          = data_feature_utils::DAAL_OTHER_T;
        pmmlType           = data_feature_utils::DAAL_GEN_UNKNOWN;
        featureType        = data_feature_utils::DAAL_CONTINUOUS;
        typeSize           = 0;
        categoryNumber     = 0;
    }

    /**
     *  Copy operator for a data feature
     */
    NumericTableFeature &operator= (const NumericTableFeature &f)
    {
        indexType          = f.indexType     ;
        pmmlType           = f.pmmlType      ;
        featureType        = f.featureType   ;
        typeSize           = f.typeSize      ;
        categoryNumber     = f.categoryNumber;

        return *this;
    }

    virtual ~NumericTableFeature() {}

    /**
     *  Fills the class based on a specified type
     *  \tparam  T  Name of the data feature
     */
    template<typename T>
    void setType()
    {
        typeSize  = sizeof(T);
        indexType = data_feature_utils::getIndexNumType<T>();
        pmmlType  = data_feature_utils::getPMMLNumType<T>();
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch)
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch)
    {serialImpl<OutputDataArchive, true>( arch );}

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->set( pmmlType         );
        arch->set( featureType      );
        arch->set( typeSize         );
        arch->set( categoryNumber   );
        arch->set( indexType        );
    }

    virtual int getSerializationTag()
    {
        return SERIALIZATION_DATAFEATURE_NT_ID;
    }
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
    void serializeImpl  (InputDataArchive  *arch)
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch)
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

    virtual int getSerializationTag()
    {
        return SERIALIZATION_DATAFEATURE_NT_ID;
    }
};

/** \private */
class DictionaryIface {};

/**
 *  <a name="DAAL-CLASS-DATADICTIONARY"></a>
 *  \brief Class that represents a dictionary of a data set
 *  and provides methods to work with the data dictionary
 */
template<typename Feature>
class Dictionary : public SerializationIface, public DictionaryIface
{
public:
    /**
     *  Constructor of a data dictionary
     *  \param[in]  nfeat  Number of features in the table
     *  \param[in]  featuresEqual Flag specifying that all features have equal types and properties
     */
    Dictionary( size_t nfeat, bool featuresEqual = false ):
        _nfeat(0), _featuresEqual(featuresEqual), _dict(0), _errors(new services::KernelErrorCollection())
    {
        if(nfeat) { setNumberOfFeatures(nfeat); }
    }

    /**
     *  Default constructor of a data dictionary
     */
    Dictionary(): _nfeat(0), _dict(0), _featuresEqual(false), _errors(new services::KernelErrorCollection()) {}

    /** \private */
    ~Dictionary()
    {
        resetDictionary();
    }

    /**
     *  Resets a dictionary and sets the number of features to 0
     */
    void resetDictionary()
    {
        if(_dict)
        {
            delete[] _dict;
            _dict = NULL;
        }
        _nfeat = 0;
    }

    /**
     *  Sets all features of a dictionary to the same type
     *  \param[in]  defaultFeature  Default feature class to which to set all features
     */
    virtual void setAllFeatures(const Feature &defaultFeature)
    {
        if (_featuresEqual)
        {
            if (_nfeat > 0)
            {
                _dict[0] = defaultFeature;
            }
        }
        else
        {
            for( size_t i = 0 ; i < _nfeat ; i++ )
            {
                _dict[i] = defaultFeature;
            }
        }
    }

    /**
     *  Sets the number of features
     *  \param[in]  numberOfFeatures  Number of features
     */
    virtual void setNumberOfFeatures(size_t numberOfFeatures)
    {
        resetDictionary();
        _nfeat = numberOfFeatures;
        if (_featuresEqual)
        {
            _dict  = new Feature[1];
        }
        else
        {
            _dict  = new Feature[_nfeat];
        }
    }

    /**
     *  Returns the number of features
     *  \return Number of features
     */
    size_t getNumberOfFeatures() const
    {
        return _nfeat;
    }

    /**
     *  Returns the value of the featuresEqual flag
     *  \return Value of the featuresEqual flag
     */
    size_t getFeaturesEqual() const
    {
        return _featuresEqual;
    }

    /**
     *  Returns a feature with a given index
     *  \param[in]  idx  Index of the feature
     *  \return Requested feature
     */
    Feature &operator[](size_t idx)
    {
        if (_featuresEqual)
        {
            return _dict[0];
        }
        else
        {
            return _dict[idx];
        }
    }

    /**
     *  \brief Adds a feature to a data dictionary
     *
     *  \param[in] feature  Data feature
     *  \param[in] idx      Index of the data feature
     *
     */
    void setFeature(const Feature &feature, size_t idx)
    {
        if(idx >= _nfeat) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (_featuresEqual)
        {
            _dict[0] = feature;
        }
        else
        {
            _dict[idx] = feature;
        }
    }

    /**
     *  Adds a feature to a data dictionary
     *  \param[in] idx              Index of the data feature
     */
    template<typename T>
    void setFeature(size_t idx)
    {
        Feature df;
        df.template setType<T>();
        setFeature(df, idx);
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     */
    services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _errors;
    }

    virtual int getSerializationTag()
    {
        if (IsSameType<Feature, NumericTableFeature>::value)
        {
            return SERIALIZATION_DATADICTIONARY_NT_ID;
        }
        else
        if (IsSameType<Feature, DataSourceFeature>::value)
        {
            return SERIALIZATION_DATADICTIONARY_DS_ID;
        }
        else
        {
            return 0;
        }
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch)
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch)
    {serialImpl<OutputDataArchive, true>( arch );}

private:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->segmentHeader();

        arch->set( _nfeat );
        arch->set( _featuresEqual );

        if( onDeserialize )
        {
            size_t nfeat = _nfeat;
            _nfeat = 0;
            setNumberOfFeatures(nfeat);
        }

        if (_featuresEqual)
        {
            arch->setObj( _dict, 1 );
        }
        else
        {
            arch->setObj( _dict, _nfeat );
        }

        arch->segmentFooter();
    }

protected:
    size_t   _nfeat;
    bool     _featuresEqual;
    Feature *_dict;
    services::SharedPtr<services::KernelErrorCollection> _errors;
};
typedef Dictionary<NumericTableFeature> NumericTableDictionary;
typedef Dictionary<DataSourceFeature>   DataSourceDictionary;
} // namespace interface1
using interface1::CategoricalFeatureDictionary;
using interface1::NumericTableFeature;
using interface1::DataSourceFeature;
using interface1::DictionaryIface;
using interface1::Dictionary;
using interface1::NumericTableDictionary;
using interface1::DataSourceDictionary;

}
} // namespace daal
#endif
