using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.OpencvAdapter.APIs
{
    public partial class Cv2API
    {
        /// <summary>
        /// Groups the object candidate rectangles.
        /// </summary>
        /// <param name="rectList"> Input/output vector of rectangles. Output vector includes retained and grouped rectangles.</param>
        /// <param name="groupThreshold">Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.</param>
        /// <param name="eps"></param>
        public void groupRectangles(IList<Rect> rectList, int groupThreshold, double eps = 0.2)
        {
            Cv2.GroupRectangles(rectList, groupThreshold, eps);
        }

        /// <summary>
        /// Groups the object candidate rectangles.
        /// </summary>
        /// <param name="rectList"></param>
        /// <param name="groupThreshold"></param>
        /// <param name="eps"></param>
        /// <param name="weights"></param>
        /// <param name="levelWeights"></param>
        public void groupRectangles(IList<Rect> rectList, int groupThreshold, double eps, out int[] weights, out double[] levelWeights)
        {
            Cv2.GroupRectangles(rectList, groupThreshold, eps, out weights, out levelWeights);
        }

        /// <summary>
        /// Groups the object candidate rectangles.
        /// </summary>
        /// <param name="rectList"></param>
        /// <param name="rejectLevels"></param>
        /// <param name="levelWeights"></param>
        /// <param name="groupThreshold"></param>
        /// <param name="eps"></param>
        public void groupRectangles(IList<Rect> rectList, out int[] rejectLevels, out double[] levelWeights, int groupThreshold, double eps = 0.2)
        {
            Cv2.GroupRectangles(rectList, out rejectLevels, out levelWeights, groupThreshold, eps);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rectList"></param>
        /// <param name="foundWeights"></param>
        /// <param name="foundScales"></param>
        /// <param name="detectThreshold"></param>
        /// <param name="winDetSize"></param>
        public static void groupRectanglesMeanshift(IList<Rect> rectList, out double[] foundWeights,
            out double[] foundScales, double detectThreshold = 0.0, Size? winDetSize = null)
        {
            Cv2.GroupRectanglesMeanshift(rectList, out foundWeights, out foundScales, detectThreshold, winDetSize);
        }
    }
}
