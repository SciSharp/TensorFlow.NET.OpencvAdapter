using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.NumPy;
using Tensorflow.OpencvAdapter.Extensions;

namespace Tensorflow.OpencvAdapter.APIs
{
    public partial class Cv2API
    {
        /// <summary>
        /// Loads an image from a file.
        /// </summary>
        /// <param name="fileName">Name of file to be loaded.</param>
        /// <param name="flags">Specifies color type of the loaded image</param>
        /// <returns></returns>
        public NDArray imread(string filename, ImreadModes flags = ImreadModes.Color)
        {
            return Cv2.ImRead(filename, flags).ToNDArray(clone: false);
        }

        /// <summary>
        /// Loads a multi-page image from a file. 
        /// </summary>
        /// <param name="filename">Name of file to be loaded.</param>
        /// <param name="mats">A vector of Mat objects holding each page, if more than one.</param>
        /// <param name="flags">Flag that can take values of @ref cv::ImreadModes, default with IMREAD_ANYCOLOR.</param>
        /// <returns></returns>
        public bool imreadmulti(string filename, out NDArray[] arrays, ImreadModes flags = ImreadModes.AnyColor)
        {
            bool res = Cv2.ImReadMulti(filename, out var mats, flags);
            arrays = mats.Select(m => m.ToNDArray(clone: false)).ToArray();
            return res;
        }

        /// <summary>
        /// Saves an image to a specified file.
        /// </summary>
        /// <param name="fileName">Name of the file.</param>
        /// <param name="img">Image to be saved.</param>
        /// <param name="prms">Format-specific save parameters encoded as pairs</param>
        /// <returns></returns>
        public bool imwrite(string filename, NDArray img, int[]? prms = null)
        {
            return Cv2.ImWrite(filename, img.AsMat(), prms);
        }

        /// <summary>
        /// Saves an image to a specified file.
        /// </summary>
        /// <param name="fileName">Name of the file.</param>
        /// <param name="img">Image to be saved.</param>
        /// <param name="prms">Format-specific save parameters encoded as pairs</param>
        /// <returns></returns>
        public bool imwrite(string filename, NDArray img, params ImageEncodingParam[] prms)
        {
            return Cv2.ImWrite(filename, img.AsMat(), prms);
        }

        /// <summary>
        /// Saves an image to a specified file.
        /// </summary>
        /// <param name="fileName">Name of the file.</param>
        /// <param name="img">Image to be saved.</param>
        /// <param name="prms">Format-specific save parameters encoded as pairs</param>
        /// <returns></returns>
        public bool imwrite(string filename, IEnumerable<NDArray> img, int[]? prms = null)
        {
            return Cv2.ImWrite(filename, img.Select(x => x.AsMat()), prms);
        }

        /// <summary>
        /// Saves an image to a specified file.
        /// </summary>
        /// <param name="fileName">Name of the file.</param>
        /// <param name="img">Image to be saved.</param>
        /// <param name="prms">Format-specific save parameters encoded as pairs</param>
        /// <returns></returns>
        public bool imwrite(string filename, IEnumerable<NDArray> img, params ImageEncodingParam[] prms)
        {
            return Cv2.ImWrite(filename, img.Select(x => x.AsMat()), prms);
        }

        /// <summary>
        /// Reads image from the specified buffer in memory.
        /// </summary>
        /// <param name="buf">The input array of vector of bytes.</param>
        /// <param name="flags">The same flags as in imread</param>
        /// <returns></returns>
        public NDArray imdecode(NDArray buf, ImreadModes flags)
        {
            return Cv2.ImDecode(buf.AsMat(), flags).ToNDArray(clone: false);
        }

        /// <summary>
        /// Reads image from the specified buffer in memory.
        /// </summary>
        /// <param name="buf">The input array of vector of bytes.</param>
        /// <param name="flags">The same flags as in imread</param>
        /// <returns></returns>
        public NDArray imdecode(byte[] buf, ImreadModes flags)
        {
            return Cv2.ImDecode(buf, flags).ToNDArray(clone: false);
        }

        /// <summary>
        /// Reads image from the specified buffer in memory.
        /// </summary>
        /// <param name="span">The input slice of bytes.</param>
        /// <param name="flags">The same flags as in imread</param>
        /// <returns></returns>
        public NDArray imdecode(ReadOnlySpan<byte> buf, ImreadModes flags)
        {
            return Cv2.ImDecode(buf, flags).ToNDArray(clone: false);
        }

        /// <summary>
        /// Compresses the image and stores it in the memory buffer
        /// </summary>
        /// <param name="ext">The file extension that defines the output format</param>
        /// <param name="img">The image to be written</param>
        /// <param name="buf">Output buffer resized to fit the compressed image.</param>
        /// <param name="prms">Format-specific parameters.</param>
        public bool imencode(string ext, NDArray img, out byte[] buf, int[]? prms = null)
        {
            return Cv2.ImEncode(ext, img.AsMat(), out buf, prms);
        }

        /// <summary>
        /// Compresses the image and stores it in the memory buffer
        /// </summary>
        /// <param name="ext">The file extension that defines the output format</param>
        /// <param name="img">The image to be written</param>
        /// <param name="buf">Output buffer resized to fit the compressed image.</param>
        /// <param name="prms">Format-specific parameters.</param>
        public void imencode(string ext, NDArray img, out byte[] buf, params ImageEncodingParam[] prms)
        {
            Cv2.ImEncode(ext, img.AsMat(), out buf, prms);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public bool haveImageReader(string filename)
        {
            return Cv2.HaveImageReader(filename);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public bool haveImageWriter(string filename)
        {
            return Cv2.HaveImageWriter(filename);
        }
    }
}
