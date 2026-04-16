using System;
using System.Collections.Generic;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Builds a <see cref="Beam"/> or <see cref="Plate"/> from a <see cref="Brep"/> by sampling geometry,
    /// computing a dominant axis via PCA, measuring a PCA-aligned bounding box, and classifying element type.
    /// </summary>
    public static class BrepElementBuilder
    {
        const int PowerIterations = 48;

        public sealed class AnalysisResult
        {
            public bool Success;
            public string Message;
            public Element Element;
            public Line Axis;
            /// <summary>Extent along first principal direction (m).</summary>
            public double LengthAlongAxis;
            /// <summary>Extents along second and third PCA frame axes (m).</summary>
            public double Extent1, Extent2;
            public Vector3d AxisDirection;
            public bool ClassifiedAsPlate;
        }

        /// <summary>
        /// Creates a beam or plate from <paramref name="brep"/> and <paramref name="material"/>.
        /// Uses <paramref name="elementId"/> / <paramref name="elementName"/> for the element; section id matches element id.
        /// </summary>
        public static AnalysisResult TryBuild(Brep brep, Material material, int elementId, string elementName, string location = "")
        {
            var result = new AnalysisResult();
            if (brep == null || !brep.IsValid)
            {
                result.Message = "Invalid Brep.";
                return result;
            }
            if (material == null)
            {
                result.Message = "Material is required.";
                return result;
            }

            if (!TrySamplePoints(brep, out List<Point3d> points, out string sampleMsg))
            {
                result.Message = sampleMsg;
                return result;
            }

            if (points.Count < 4)
            {
                result.Message = "Too few sample points for PCA.";
                return result;
            }

            ComputeCovariance(points, out Point3d centroid, out double[,] cov);

            if (!TryPrincipalAxis(cov, out Vector3d e0))
            {
                result.Message = "Could not compute principal axis (degenerate covariance).";
                return result;
            }

            BuildOrthonormalFrame(e0, out e0, out Vector3d e1, out Vector3d e2);

            AxisExtents(points, centroid, e0, e1, e2,
                out double L, out double w, out double h,
                out Line axisLine);

            if (L < Rhino.RhinoMath.ZeroTolerance)
            {
                result.Message = "Zero length along PCA axis.";
                return result;
            }

            bool geomPlate = ClassifyGeometryPlate(L, w, h);
            bool geomBeam  = ClassifyGeometryBeam(L, w, h);

            bool usePlate = geomPlate;
            if (geomPlate && geomBeam)
                usePlate = TieBreakWithMaterial(material.Name, L, w, h);
            else if (!geomPlate && !geomBeam)
                usePlate = ElementClassifier.ClassifyByGeometry(w, h, L, false);

            result.Axis = axisLine;
            result.AxisDirection = e0;
            result.LengthAlongAxis = L;
            result.Extent1 = w;
            result.Extent2 = h;
            result.ClassifiedAsPlate = usePlate;

            string safeName = string.IsNullOrWhiteSpace(elementName)
                ? (usePlate ? "Plate_FromBrep" : "Beam_FromBrep")
                : elementName;

            if (usePlate)
            {
                double thickness = Math.Min(w, h);
                double panelW    = Math.Max(w, h);
                var section = new PlateSection(elementId, $"T{thickness * 1000.0:0}mm", thickness, panelW);
                var plate = new Plate(elementId, safeName, location ?? "", material, section, null)
                {
                    AxisLine   = axisLine,
                    GeometryBrep = brep.DuplicateBrep()
                };
                result.Element = plate;
                result.Success = true;
                result.Message =
                    $"Plate: L={L:F4} m, T={thickness:F4} m, W={panelW:F4} m (PCA extents w×h={w:F4}×{h:F4}).";
            }
            else
            {
                var section = new BeamSection(elementId, $"{w * 1000.0:0}x{h * 1000.0:0}", w, h);
                var beam = new Beam(elementId, safeName, location ?? "", material, section, axisLine)
                {
                    GeometryBrep = brep.DuplicateBrep()
                };
                result.Element = beam;
                result.Success = true;
                result.Message =
                    $"Beam: L={L:F4} m, section {w:F4}×{h:F4} m (along PCA axis).";
            }

            return result;
        }

        // ── sampling ───────────────────────────────────────────────────────

        static bool TrySamplePoints(Brep brep, out List<Point3d> points, out string message)
        {
            points = null;
            message = null;
            try
            {
                var mp = MeshingParameters.FastRenderMesh;
                var box = brep.GetBoundingBox(true);
                double diag = box.Diagonal.Length;
                if (diag > 1e-9)
                {
                    mp.MinimumEdgeLength = Math.Max(diag * 0.015, 1e-4);
                    mp.MaximumEdgeLength = Math.Max(diag * 0.12, mp.MinimumEdgeLength * 2);
                }

                var meshes = Mesh.CreateFromBrep(brep, mp);
                if (meshes == null || meshes.Length == 0)
                {
                    message = "Mesh.CreateFromBrep returned no mesh.";
                    return false;
                }

                points = new List<Point3d>(4096);
                foreach (var m in meshes)
                {
                    if (m == null) continue;
                    for (int i = 0; i < m.Vertices.Count; i++)
                        points.Add(m.Vertices[i]);
                }

                if (points.Count < 4)
                {
                    message = "Too few mesh vertices.";
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                return false;
            }
        }

        // ── PCA ───────────────────────────────────────────────────────────

        static void ComputeCovariance(List<Point3d> pts, out Point3d centroid, out double[,] c)
        {
            int n = pts.Count;
            double sx = 0, sy = 0, sz = 0;
            foreach (var p in pts)
            {
                sx += p.X; sy += p.Y; sz += p.Z;
            }
            centroid = new Point3d(sx / n, sy / n, sz / n);

            c = new double[3, 3];
            double invNm1 = 1.0 / Math.Max(n - 1, 1);
            foreach (var p in pts)
            {
                double dx = p.X - centroid.X;
                double dy = p.Y - centroid.Y;
                double dz = p.Z - centroid.Z;
                c[0, 0] += dx * dx * invNm1;
                c[0, 1] += dx * dy * invNm1;
                c[0, 2] += dx * dz * invNm1;
                c[1, 1] += dy * dy * invNm1;
                c[1, 2] += dy * dz * invNm1;
                c[2, 2] += dz * dz * invNm1;
            }
            c[1, 0] = c[0, 1];
            c[2, 0] = c[0, 2];
            c[2, 1] = c[1, 2];
        }

        static bool TryPrincipalAxis(double[,] c, out Vector3d axis)
        {
            axis = Vector3d.Zero;
            double x = 1, y = 0, z = 0;
            for (int it = 0; it < PowerIterations; it++)
            {
                double nx = c[0, 0] * x + c[0, 1] * y + c[0, 2] * z;
                double ny = c[1, 0] * x + c[1, 1] * y + c[1, 2] * z;
                double nz = c[2, 0] * x + c[2, 1] * y + c[2, 2] * z;
                double len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                if (len < 1e-18) return false;
                x = nx / len; y = ny / len; z = nz / len;
            }
            axis = new Vector3d(x, y, z);
            if (!axis.Unitize()) return false;
            return true;
        }

        static void BuildOrthonormalFrame(Vector3d e0, out Vector3d axis, out Vector3d e1, out Vector3d e2)
        {
            axis = e0;
            axis.Unitize();
            Vector3d refDir = Math.Abs(axis * Vector3d.ZAxis) > 0.9 ? Vector3d.XAxis : Vector3d.ZAxis;
            e1 = Vector3d.CrossProduct(refDir, axis);
            if (e1.Length < 1e-12)
                e1 = Vector3d.CrossProduct(Vector3d.YAxis, axis);
            e1.Unitize();
            e2 = Vector3d.CrossProduct(axis, e1);
            e2.Unitize();
        }

        static void AxisExtents(List<Point3d> pts, Point3d centroid,
            Vector3d e0, Vector3d e1, Vector3d e2,
            out double L, out double w, out double h, out Line axisLine)
        {
            double min0 = double.MaxValue, max0 = double.MinValue;
            double min1 = double.MaxValue, max1 = double.MinValue;
            double min2 = double.MaxValue, max2 = double.MinValue;

            foreach (var p in pts)
            {
                Vector3d r = p - centroid;
                double s0 = r * e0;
                double s1 = r * e1;
                double s2 = r * e2;
                if (s0 < min0) min0 = s0;
                if (s0 > max0) max0 = s0;
                if (s1 < min1) min1 = s1;
                if (s1 > max1) max1 = s1;
                if (s2 < min2) min2 = s2;
                if (s2 > max2) max2 = s2;
            }

            L = Math.Max(max0 - min0, 0);
            w = Math.Max(max1 - min1, 0);
            h = Math.Max(max2 - min2, 0);

            var a = centroid + e0 * min0;
            var b = centroid + e0 * max0;
            axisLine = new Line(a, b);
        }

        // ── classification ────────────────────────────────────────────────

        /// <summary>Thin transverse direction or stocky cross-section vs length → plate.</summary>
        static bool ClassifyGeometryPlate(double L, double w, double h)
        {
            if (L < 1e-9) return false;
            double maxCross = Math.Max(w, h);
            double minCross = Math.Min(w, h);

            if (minCross / L < 0.42)
                return true;

            if (maxCross > L / 6.5)
                return true;

            return false;
        }

        /// <summary>Slender member along PCA axis → beam.</summary>
        static bool ClassifyGeometryBeam(double L, double w, double h)
        {
            double maxCross = Math.Max(w, h);
            if (maxCross < 1e-9) return false;
            if (L / maxCross >= 2.15)
                return true;
            return false;
        }

        static bool TieBreakWithMaterial(string materialName, double L, double w, double h)
        {
            return ElementClassifier.IsPlate(materialName ?? "", w, h, L, false);
        }
    }
}
