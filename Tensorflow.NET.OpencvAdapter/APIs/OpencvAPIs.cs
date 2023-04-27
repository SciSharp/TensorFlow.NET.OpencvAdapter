using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.OpencvAdapter.APIs;

namespace Tensorflow
{
    public class OpencvAPIs
    {
        public static Cv2API cv2 { get; } = new Cv2API();
    }
}
