using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.OpencvAdapter
{
    internal static class AdapterUtils
    {
        private static readonly IReadOnlyDictionary<MatType, TF_DataType> _matTypeMappingToTFDataType = new Dictionary<MatType, TF_DataType>
        {
            [MatType.CV_8UC1] = TF_DataType.TF_UINT8,
            [MatType.CV_8SC1] = TF_DataType.TF_INT8,
            [MatType.CV_16SC1] = TF_DataType.TF_INT16,
            [MatType.CV_16UC1] = TF_DataType.TF_UINT16,
            [MatType.CV_32SC1] = TF_DataType.TF_INT32,
            [MatType.CV_32FC1] = TF_DataType.TF_FLOAT,
            [MatType.CV_64FC1] = TF_DataType.TF_DOUBLE,

            [MatType.CV_8UC2] = TF_DataType.TF_UINT8,
            [MatType.CV_8UC3] = TF_DataType.TF_UINT8,
            [MatType.CV_8UC4] = TF_DataType.TF_UINT8,

            [MatType.CV_16SC2] = TF_DataType.TF_INT16,
            [MatType.CV_16SC3] = TF_DataType.TF_INT16,
            [MatType.CV_16SC4] = TF_DataType.TF_INT16,

            [MatType.CV_16UC2] = TF_DataType.TF_UINT16,
            [MatType.CV_16UC3] = TF_DataType.TF_UINT16,
            [MatType.CV_16UC4] = TF_DataType.TF_UINT16,

            [MatType.CV_32SC2] = TF_DataType.TF_INT32,
            [MatType.CV_32SC3] = TF_DataType.TF_INT32,
            [MatType.CV_32SC4] = TF_DataType.TF_INT32,

            [MatType.CV_32FC2] = TF_DataType.TF_FLOAT,
            [MatType.CV_32FC3] = TF_DataType.TF_FLOAT,
            [MatType.CV_32FC4] = TF_DataType.TF_FLOAT,

            [MatType.CV_64FC2] = TF_DataType.TF_DOUBLE,
            [MatType.CV_64FC3] = TF_DataType.TF_DOUBLE,
            [MatType.CV_64FC4] = TF_DataType.TF_DOUBLE,
        };

        internal static TF_DataType MatTypeToTFDataType(MatType type)
        {
            if(_matTypeMappingToTFDataType.TryGetValue(type, out var res))
            {
                return res;
            }
            else
            {
                throw new TypeError($"MatType {Enum.GetName(typeof(MatType), type)} is invalid " +
                    $"or is not supported in tensorflow opencv adapter. The developers of tensorflow " +
                    $"opencv adapter cannot decide which types to support at the beginning so that only some " +
                    $"basic types is supported. For example, the using of vec4 and vec6 seems to be rare. " +
                    $"Please submit an issue to tell us the condition of this MatType " +
                    $"and we will add support for it");
            }
        }

        internal static MatType TFDataTypeToMatType(TF_DataType type, int channels)
        {
            if(channels == 1)
            {
                return type switch
                {
                    TF_DataType.TF_UINT8 => MatType.CV_8UC1,
                    TF_DataType.TF_UINT16 => MatType.CV_16UC1,
                    TF_DataType.TF_INT8 => MatType.CV_8SC1,
                    TF_DataType.TF_INT16 => MatType.CV_16SC1,
                    TF_DataType.TF_INT32 => MatType.CV_32SC1,
                    TF_DataType.TF_FLOAT => MatType.CV_32FC1,
                    TF_DataType.TF_DOUBLE => MatType.CV_64FC1
                };
            }
            else if(channels == 2)
            {
                return type switch
                {
                    TF_DataType.TF_UINT8 => MatType.CV_8UC2,
                    TF_DataType.TF_UINT16 => MatType.CV_16UC2,
                    TF_DataType.TF_INT8 => MatType.CV_8SC2,
                    TF_DataType.TF_INT16 => MatType.CV_16SC2,
                    TF_DataType.TF_INT32 => MatType.CV_32SC2,
                    TF_DataType.TF_FLOAT => MatType.CV_32FC2,
                    TF_DataType.TF_DOUBLE => MatType.CV_64FC2
                };
            }
            else if(channels == 3)
            {
                return type switch
                {
                    TF_DataType.TF_UINT8 => MatType.CV_8UC3,
                    TF_DataType.TF_UINT16 => MatType.CV_16UC3,
                    TF_DataType.TF_INT8 => MatType.CV_8SC3,
                    TF_DataType.TF_INT16 => MatType.CV_16SC3,
                    TF_DataType.TF_INT32 => MatType.CV_32SC3,
                    TF_DataType.TF_FLOAT => MatType.CV_32FC3,
                    TF_DataType.TF_DOUBLE => MatType.CV_64FC3
                };
            }
            else if(channels == 4)
            {
                return type switch
                {
                    TF_DataType.TF_UINT8 => MatType.CV_8UC4,
                    TF_DataType.TF_UINT16 => MatType.CV_16UC4,
                    TF_DataType.TF_INT8 => MatType.CV_8SC4,
                    TF_DataType.TF_INT16 => MatType.CV_16SC4,
                    TF_DataType.TF_INT32 => MatType.CV_32SC4,
                    TF_DataType.TF_FLOAT => MatType.CV_32FC4,
                    TF_DataType.TF_DOUBLE => MatType.CV_64FC4
                };
            }
            else
            {
                throw new ValueError($"{channels} channels data is not supported by tensorflow.net opencv adapter. " +
                    $"If you think it's an expected behavior, please submit an issue to tell us.");
            }
        }
        
        internal static void SetMatFromNDArrayData(NDArray array, Mat mat)
        {
            if(array.dtype == TF_DataType.TF_FLOAT)
            {
                mat.SetArray(array.ToArray<float>());
            }
            else if(array.dtype == TF_DataType.TF_DOUBLE)
            {
                mat.SetArray(array.ToArray<double>());
            }
            else if(array.dtype == TF_DataType.TF_INT32)
            {
                mat.SetArray(array.ToArray<int>());
            }
            else if(array.dtype == TF_DataType.TF_INT16)
            {
                mat.SetArray(array.ToArray<short>());
            }
            else if(array.dtype == TF_DataType.TF_INT8)
            {
                mat.SetArray(array.ToArray<sbyte>());
            }
            else if(array.dtype == TF_DataType.TF_UINT16)
            {
                mat.SetArray(array.ToArray<ushort>());
            }
            else if(array.dtype == TF_DataType.TF_UINT8)
            {
                mat.SetArray(array.ToArray<byte>());
            }
            else
            {
                throw new ValueError($"Type {array.dtype.as_numpy_name()} is not supported to convert to Mat.");
            }
        }

        /// <summary>
        /// The layout should be "hwc"
        /// </summary>
        /// <param name="array"></param>
       internal static Mat ConvertNDArrayToMat(NDArray array)
        {
            if (CvNDArray.AdapterMode == OpencvAdapterMode.StrictNoCopy || CvNDArray.AdapterMode == OpencvAdapterMode.AllowCopy)
            {
                var dataPointer = array.TensorDataPointer;
                var (matType, rows, cols) = DeduceMatInfoFromNDArray(array.shape, array.dtype);
                return new Mat(rows, cols, matType, dataPointer);
            }
            else // AdapterMode == OpencvAdapterMode.AlwaysCopy
            {
                var (matType, rows, cols) = DeduceMatInfoFromNDArray(array.shape, array.dtype);
                Mat m = new Mat(rows, cols, matType);
                SetMatFromNDArrayData(array, m);
                return m;
            }
        }

        internal static (MatType, int, int) DeduceMatInfoFromNDArray(Shape shape, TF_DataType dtype)
        {
            if (shape.rank <= 1 || shape.rank >= 4)
            {
                throw new ValueError($"Converting from NDArray to Mat with shape with rank {shape.rank} has not been supported. If it's expected to work with you, " +
                    $"please submit an issue and we'll add it.");
            }
            if (shape[0] > int.MaxValue || shape[1] > int.MaxValue)
            {
                throw new ValueError($"The shape {shape} is too large to convert to CvNDArray");
            }
            int rows = (int)shape[0];
            int cols = (int)shape[1];
            MatType matType;
            if (shape.rank == 2)
            {
                matType = TFDataTypeToMatType(dtype, 1);
            }
            else // shape.rank == 3
            {
                matType = TFDataTypeToMatType(dtype, (int)shape[2]);
            }
            return (matType, rows, cols);
        }
    }
}
