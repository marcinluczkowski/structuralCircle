using System;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Decides whether a timber row should be modelled as a Plate (CLT panel) or a Beam (glulam).
    /// Priority: material-name keywords → geometric slenderness rule.
    /// </summary>
    public static class ElementClassifier
    {
        public static bool IsPlate(string materialType, double width, double height, double lengthMetres,
            bool widthHeightInMillimetres)
        {
            string u = (materialType ?? "").ToUpperInvariant();

            if (HasPlateMaterialKeywords(u)) return true;
            if (HasBeamMaterialKeywords(u))  return false;

            return ClassifyByGeometry(width, height, lengthMetres, widthHeightInMillimetres);
        }

        /// <summary>
        /// max(W,H) > L/7  →  plate-like panel; otherwise slender beam.
        /// Dimensions may be in mm or m; lengthMetres always in metres.
        /// </summary>
        public static bool ClassifyByGeometry(double width, double height, double lengthMetres,
            bool widthHeightInMillimetres)
        {
            double wmm = widthHeightInMillimetres ? width  : width  * 1000.0;
            double hmm = widthHeightInMillimetres ? height : height * 1000.0;
            double Lmm = Math.Max(lengthMetres * 1000.0, 1.0);

            double maxCross = Math.Max(wmm, hmm);
            return maxCross > Lmm / 7.0;
        }

        static bool HasPlateMaterialKeywords(string u)
        {
            if (u.Contains("X-LAM") || u.Contains("XLAM")) return true;
            if (u.Contains("CLT"))                          return true;
            if (u.Contains("CROSS") && u.Contains("LAM"))  return true;
            if (u.Contains("CROSS-LAMINAT"))               return true;
            if (u.Contains("KERTO-Q"))                     return true;
            return false;
        }

        static bool HasBeamMaterialKeywords(string u)
        {
            if (u.Contains("LIMTRE"))                                   return true;
            if (u.Contains("GLULAM"))                                   return true;
            if (u.Contains("GL24") || u.Contains("GL28") || u.Contains("GL32")) return true;
            if (u.Contains("KVH"))                                      return true;
            if (u.Contains("BSH") || u.Contains("BRETT"))               return true;
            return false;
        }
    }
}
