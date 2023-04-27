using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.OpencvAdapter.Extensions
{
    internal static class DTypeExtensions
    {
        internal static int ToMatTypeNumber(this TF_DataType dtype, int channels)
        {
            if(dtype == TF_DataType.DtInvalid)
            {
                return -1;
            }
            else
            {
                return AdapterUtils.TFDataTypeToMatType(dtype, channels);
            }
        }
    }
}
