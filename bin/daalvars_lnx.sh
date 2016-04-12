#!/bin/bash
#===============================================================================
# Copyright 2014-2016 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

daal_help() {
    echo "Syntax: source daalvars.sh <arch>"
    echo "Where <arch> is one of:"
    echo "  ia32      - setup environment for IA-32 architecture"
    echo "  intel64   - setup environment for Intel(R) 64 architecture"
}

set_daal_env() {
    __daal_tmp_dir="<INSTALLDIR>"
    __daal_tmp_dir=$__daal_tmp_dir/daal
    if [ ! -d $__daal_tmp_dir ]; then
        __daal_tmp_dir=$(command -p cd $(dirname -- "${BASH_SOURCE}")/..; pwd)
    fi

    if [ -z "$1" ] ; then
        if [ -n "$DAALVARS_ARCHITECTURE" ] ; then
            ia="$DAALVARS_ARCHITECTURE"
        elif [ -n "$COMPILERVARS_ARCHITECTURE" ] ; then
            ia="$COMPILERVARS_ARCHITECTURE"
        fi
    else
        ia=$1
    fi

    case $ia in
        ia32|intel64)
            export DAALROOT=$__daal_tmp_dir
            export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
            if [ -z "$TBBROOT" ]; then
                export LIBRARY_PATH=$__daal_tmp_dir/lib/${ia}_lin:$__daal_tmp_dir/../tbb/lib/${ia}_lin/gcc4.4${LIBRARY_PATH+:${LIBRARY_PATH}}
                export LD_LIBRARY_PATH=$__daal_tmp_dir/lib/${ia}_lin:$__daal_tmp_dir/../tbb/lib/${ia}_lin/gcc4.4${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}
            else
                export LIBRARY_PATH=$__daal_tmp_dir/lib/${ia}_lin${LIBRARY_PATH+:${LIBRARY_PATH}}
                export LD_LIBRARY_PATH=$__daal_tmp_dir/lib/${ia}_lin${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}
            fi
            export CLASSPATH=$__daal_tmp_dir/lib/daal.jar${CLASSPATH+:${CLASSPATH}}
            ;;
        *) daal_help
            ;;
    esac
}

set_daal_env "$@"
