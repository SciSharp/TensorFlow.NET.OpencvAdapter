using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.OpencvAPIs;

namespace Tensorflow.OpencvAdapter.Unittest
{
    [TestClass]
    public class ImageCodecsTest
    {
        [TestMethod]
        public void LoadImageAndSave()
        {
            string filename = "Assets/test1.JPEG";
            var img = cv2.imread(filename);
            Console.WriteLine(img.ToString());

            Assert.AreEqual(17, (int)img[0, 0, 0]);
            Assert.AreEqual(184, (int)img[0, 0, 1]);
            Assert.AreEqual(197, (int)img[0, 0, 2]);
            Assert.AreEqual(13, (int)img[0, 1, 0]);
            Assert.AreEqual(181, (int)img[0, 1, 1]);
            Assert.AreEqual(192, (int)img[0, 1, 2]);

            cv2.imwrite("Assets/saved_test1.jpg", cv2.subtract(img, Scalar.FromDouble(1.0)));
        }
    }
}
