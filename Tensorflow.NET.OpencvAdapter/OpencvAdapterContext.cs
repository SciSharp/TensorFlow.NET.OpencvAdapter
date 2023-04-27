using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.OpencvAdapter
{
    public class OpencvAdapterContext: IDisposable
    {
        OpencvAdapterMode _oldMode;
        public OpencvAdapterContext(OpencvAdapterMode mode)
        {
            _oldMode = CvNDArray.AdapterMode;
            CvNDArray.AdapterMode = mode;
        }
        public void Dispose()
        {
            CvNDArray.AdapterMode = _oldMode;
        }
    }
}
