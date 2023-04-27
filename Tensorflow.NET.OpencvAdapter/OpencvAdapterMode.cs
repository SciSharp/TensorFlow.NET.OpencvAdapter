using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.OpencvAdapter
{
    /// <summary>
    /// Decide the behavior of CvNDArray, which combines Mat and NDArray to provide APIs.
    /// </summary>
    public enum OpencvAdapterMode
    {
        /// <summary>
        /// Memory copying in the adapter is never allowed. This mode minimizes the cost of memory copying.
        /// However, sometimes the memory sharing between NDArray and Mat are not allowed because the mat 
        /// is not contiguous. In this case under this mode, an exception will be thrown.
        /// </summary>
        StrictNoCopy, 
        /// <summary>
        /// Memory copying is allowed when necessary. If the mat is not contiguous, then a new Mat will 
        /// be cloned for data sharing.
        /// </summary>
        AllowCopy,
        /// <summary>
        /// Momery copying is always done. Under this mode, the mat data will be copied every timr when a
        /// CvNDarray is created.
        /// </summary>
        AlwaysCopy
    }
}
