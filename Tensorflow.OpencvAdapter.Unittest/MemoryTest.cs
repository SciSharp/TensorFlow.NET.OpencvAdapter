using OpenCvSharp;
using Tensorflow.NumPy;
using Tensorflow.OpencvAdapter.Extensions;

namespace Tensorflow.OpencvAdapter.Unittest
{
    [TestClass]
    public class MemoryTest
    {
        [TestMethod]
        public void BasicUsage()
        {
            var img = Cv2.ImRead(@"Assets/test1.JPEG");
            var n = new CvNDArray(img);
            img.Set<byte>(0, 0, 111);
            Assert.AreEqual(111, (byte)n[0, 0, 0]);
        }

        [TestMethod]
        public void MemoryRelease()
        {
            var img = Cv2.ImRead(@"Assets/test1.JPEG");
            var n = new CvNDArray(img);
            n.Dispose();
            GC.Collect();
            Assert.ThrowsException<ObjectDisposedException>(() => { img.CvPtr.ToString(); });
        }

        [TestMethod]
        public void MatFromNDArray()
        {
            var array = np.load(@"Assets/img.npy");
            Mat m = array.ToMat(clone: false);
            m.Set<byte>(5, 6, 111);
            Assert.AreEqual(111, (byte)array[5, 6, 0]);
            Cv2.ImWrite(@"Assets/img.jpg", m);

            m.Release();
            GC.Collect();
            Assert.AreEqual(111, (byte)array[5, 6, 0]);
        }
    }
}