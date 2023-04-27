using OpenCvSharp;
using OpenCvSharp.Internal;
using System;
using Tensorflow.NumPy;

namespace Tensorflow.OpencvAdapter
{
    public class CvNDArray: NDArray
    {
        public static OpencvAdapterMode AdapterMode { get; set; } = OpencvAdapterMode.AllowCopy;
        protected Mat _mat;
        public unsafe CvNDArray(Mat mat)
        {
            // If mode is AlwaysCopy, then just copy it.
            if(AdapterMode == OpencvAdapterMode.AlwaysCopy)
            {
                _mat = mat.Clone();
            }
            // If the  mat is not contiguous, then a memory copy will happen to get a contiguous mat.
            else if(!mat.IsContinuous())
            {
                if(AdapterMode == OpencvAdapterMode.AllowCopy)
                {
                    _mat = mat.Clone();
                }
                else if(AdapterMode == OpencvAdapterMode.StrictNoCopy)
                {
                    throw new RuntimeError($"The CvNDarray cannot be constructed because the mat is not " +
                        $"contiguous and the mode is set to `StrictNoCopy`. Please consider changing the mode or " +
                        $"avoiding incontiguous Mat.");
                }
                else
                {
                    throw new ValueError($"Cannot recognize the mode {Enum.GetName(typeof(OpencvAdapterMode), AdapterMode)}");
                }
            }
            else
            {
                _mat = mat;
            }
            InitWithExistingMemory(new IntPtr(_mat.DataPointer), new Shape(_mat.Rows, _mat.Cols, _mat.Channels()),
                AdapterUtils.MatTypeToTFDataType(_mat.Type()), (x, y, z) => { if(_mat is not null) _mat.Release(); _mat = null; });
        }

        public Mat AsMat()
        {
            return _mat;
        }

        public override string ToString()
        {
            return "NDarray which shares memory with Mat: " + base.ToString();
        }

        protected override void DisposeManagedResources()
        {
            if(_mat is not null)
            {
                _mat.Release();
                _mat = null;
            }
            base.DisposeManagedResources();
        }
    }
}
