using System;
using System.Collections.Generic;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Builds a <see cref="Beam"/> or <see cref="Plate"/> from a <see cref="Brep"/> by sampling surface points
    /// (area-weighted count per face, random triangle or UV), computing a dominant axis via PCA,
    /// measuring a PCA-aligned bounding box, and classifying element type.
    /// </summary>
    public static class BrepElementBuilder
    {
        const int PowerIterations = 48;

        /// <summary>Total number of random surface samples on the <see cref="Brep"/>, split across faces by face area.</summary>
        public const int TargetSurfaceSampleCount = 500;

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
        /// Computes the structural axis line from a Brep using the same PCA pipeline as <see cref="TryBuild"/>:
        /// surface sampling, covariance, power iteration for the first principal direction, then axis endpoints from OBB along e0.
        /// </summary>
        public static bool TryGetPcaAxis(Brep brep, out Line axis, out string message)
            => TryGetPcaAxis(brep, out axis, out message, out _);

        /// <inheritdoc cref="TryGetPcaAxis(Brep, out Line, out string)"/>
        /// <param name="samplePoints">Surface sample points used as PCA inputs (same list as the algorithm uses).</param>
        public static bool TryGetPcaAxis(Brep brep, out Line axis, out string message, out List<Point3d> samplePoints)
        {
            axis = Line.Unset;
            message = null;
            samplePoints = null;
            if (!TryComputePcaGeometry(brep, out axis, out _, out _, out _, out _, out message, out samplePoints, out _))
                return false;
            return true;
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

            if (!TryComputePcaGeometry(brep, out Line axisLine, out double L, out double w, out double h, out Vector3d e0, out string geomMsg, out _, out _))
            {
                result.Message = geomMsg;
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

        /// <summary>Shared PCA + OBB axis line (used by <see cref="TryGetPcaAxis"/> and <see cref="TryBuild"/>).</summary>
        static bool TryComputePcaGeometry(Brep brep, out Line axisLine, out double L, out double w, out double h,
            out Vector3d axisDirection, out string message, out List<Point3d> samplePoints, out Box pcaFirstAxisBox)
        {
            axisLine = Line.Unset;
            L = w = h = 0;
            axisDirection = Vector3d.Zero;
            message = null;
            samplePoints = null;
            pcaFirstAxisBox = Box.Unset;

            if (brep == null || !brep.IsValid)
            {
                message = "Invalid Brep.";
                return false;
            }

            if (!TrySamplePoints(brep, out List<Point3d> points, out string sampleMsg))
            {
                message = sampleMsg;
                return false;
            }

            samplePoints = points;

            if (points.Count < 4)
            {
                message = "Too few sample points for PCA.";
                return false;
            }

            ComputeCovariance(points, out Point3d centroid, out double[,] cov);

            if (!TryPrincipalAxis(cov, out Vector3d e0))
            {
                message = "Could not compute principal axis (degenerate covariance).";
                return false;
            }

            BuildOrthonormalFrame(e0, out e0, out Vector3d e1, out Vector3d e2);

            AxisExtents(points, centroid, e0, e1, e2, out L, out w, out h, out axisLine, out pcaFirstAxisBox);
            axisDirection = e0;

            if (L < Rhino.RhinoMath.ZeroTolerance)
            {
                message = "Zero length along PCA axis.";
                axisLine = Line.Unset;
                pcaFirstAxisBox = Box.Unset;
                return false;
            }

            return true;
        }

        /// <summary>Sample points used for PCA / OBB (area-weighted random surface points; edge fallback if needed).</summary>
        public static bool TryGetSamplePoints(Brep brep, out List<Point3d> points, out string message)
            => TrySamplePoints(brep, out points, out message);

        /// <summary>
        /// PCA member axis and a right-handed orthonormal frame (e0 = axis, e1/e2 span cross-section planes).
        /// </summary>
        public static bool TryGetPcaAxisAndFrame(Brep brep, out Line axis, out Vector3d e0, out Vector3d e1, out Vector3d e2,
            out List<Point3d> samples, out string message)
        {
            axis = Line.Unset;
            e0 = e1 = e2 = Vector3d.Zero;
            samples = null;
            message = null;
            if (!TryComputePcaGeometry(brep, out axis, out _, out _, out _, out e0, out message, out samples, out _))
                return false;
            BuildOrthonormalFrame(e0, out e0, out e1, out e2);
            return axis.IsValid;
        }

        /// <summary>Oriented box: first axis = dominant PCA direction; other two axes from <see cref="BuildOrthonormalFrame"/>.</summary>
        public static bool TryGetPcaFirstAxisAlignedBox(Brep brep, out Box box, out string message)
        {
            box = Box.Unset;
            message = null;
            if (!TryComputePcaGeometry(brep, out _, out _, out _, out _, out _, out message, out _, out box))
                return false;
            return box.IsValid;
        }

        /// <summary>
        /// Heuristic minimum-volume oriented box from the same point samples used for PCA.
        /// Tries world axes, PCA frames, in-plane rotations, and pseudo-random orientations; not a globally optimal 3D OBB.
        /// </summary>
        public static bool TryMinimumVolumeOrientedBoxFromPoints(List<Point3d> points, out Box box, out string message)
        {
            box = Box.Unset;
            message = null;
            if (points == null || points.Count < 4)
            {
                message = "Too few sample points.";
                return false;
            }

            ComputeCovariance(points, out Point3d centroid, out double[,] cov);
            if (!TryPrincipalAxis(cov, out Vector3d e0))
            {
                message = "Could not compute principal axis (degenerate covariance).";
                return false;
            }

            BuildOrthonormalFrame(e0, out e0, out Vector3d e1, out Vector3d e2);
            bool haveFullPca = TrySecondPrincipalDirection(cov, e0, out Vector3d e1f, out Vector3d e2f);

            double bestVol = double.MaxValue;
            Box best = Box.Unset;

            void Consider(Vector3d u, Vector3d v, Vector3d w)
            {
                if (!TryUnitizeOrthonormal(ref u, ref v, ref w))
                    return;
                ProjectExtents(points, centroid, u, v, w,
                    out double min0, out double max0, out double min1, out double max1, out double min2, out double max2);
                double dx = max0 - min0, dy = max1 - min1, dz = max2 - min2;
                if (dx <= Rhino.RhinoMath.ZeroTolerance || dy <= Rhino.RhinoMath.ZeroTolerance ||
                    dz <= Rhino.RhinoMath.ZeroTolerance)
                    return;
                double vol = dx * dy * dz;
                if (vol < bestVol)
                {
                    bestVol = vol;
                    best = BuildBoxFromObbExtents(centroid, u, v, w, min0, max0, min1, max1, min2, max2);
                }
            }

            Consider(Vector3d.XAxis, Vector3d.YAxis, Vector3d.ZAxis);
            Consider(Vector3d.YAxis, Vector3d.ZAxis, Vector3d.XAxis);
            Consider(Vector3d.ZAxis, Vector3d.XAxis, Vector3d.YAxis);
            Consider(e0, e1, e2);
            if (haveFullPca)
                Consider(e0, e1f, e2f);

            const int aroundAxisSteps = 36;
            for (int k = 0; k < aroundAxisSteps; k++)
            {
                double t = k * (Math.PI / 2.0) / aroundAxisSteps;
                double ct = Math.Cos(t), st = Math.Sin(t);
                var a1 = e1 * ct + e2 * st;
                var a2 = e1 * (-st) + e2 * ct;
                Consider(e0, a1, a2);
            }

            if (haveFullPca)
            {
                for (int k = 0; k < aroundAxisSteps; k++)
                {
                    double t = k * (Math.PI / 2.0) / aroundAxisSteps;
                    double ct = Math.Cos(t), st = Math.Sin(t);
                    var b1 = e1f * ct + e2f * st;
                    var b2 = e1f * (-st) + e2f * ct;
                    Consider(e0, b1, b2);
                }
            }

            for (int k = 0; k < aroundAxisSteps; k++)
            {
                double t = k * (Math.PI / 2.0) / aroundAxisSteps;
                double ct = Math.Cos(t), st = Math.Sin(t);
                var c1 = e0 * ct + e1 * st;
                var c2 = e0 * (-st) + e1 * ct;
                Consider(c1, c2, e2);
            }

            int seed = unchecked(points.Count * 486187739 + 991997);
            var rng = new Random(seed);
            for (int r = 0; r < 96; r++)
            {
                double ax = (rng.NextDouble() - 0.5) * Math.PI;
                double ay = (rng.NextDouble() - 0.5) * Math.PI;
                double az = (rng.NextDouble() - 0.5) * Math.PI;
                var T = Transform.Rotation(az, Vector3d.ZAxis, Point3d.Origin)
                        * Transform.Rotation(ay, Vector3d.YAxis, Point3d.Origin)
                        * Transform.Rotation(ax, Vector3d.XAxis, Point3d.Origin);
                var u = new Vector3d(1, 0, 0);
                var v = new Vector3d(0, 1, 0);
                var w = new Vector3d(0, 0, 1);
                u.Transform(T);
                v.Transform(T);
                w.Transform(T);
                Consider(u, v, w);
            }

            if (!best.IsValid)
            {
                message = "Could not find a valid oriented box.";
                return false;
            }

            box = best;
            return true;
        }

        /// <summary>Same as <see cref="TryGetPcaFirstAxisAlignedBox"/> but reuses an existing sample set.</summary>
        internal static bool TryPcaFirstAxisBoxFromPoints(List<Point3d> points, out Box box, out string message)
        {
            box = Box.Unset;
            message = null;
            if (points == null || points.Count < 4)
            {
                message = "Too few sample points.";
                return false;
            }

            ComputeCovariance(points, out Point3d centroid, out double[,] cov);
            if (!TryPrincipalAxis(cov, out Vector3d e0))
            {
                message = "Could not compute principal axis (degenerate covariance).";
                return false;
            }

            BuildOrthonormalFrame(e0, out e0, out Vector3d e1, out Vector3d e2);
            AxisExtents(points, centroid, e0, e1, e2, out double L, out _, out _, out _, out box);
            if (L < Rhino.RhinoMath.ZeroTolerance)
            {
                message = "Zero length along PCA axis.";
                box = Box.Unset;
                return false;
            }

            return box.IsValid;
        }

        static Box BuildBoxFromObbExtents(Point3d centroid, Vector3d e0, Vector3d e1, Vector3d e2,
            double min0, double max0, double min1, double max1, double min2, double max2)
        {
            Point3d c = centroid
                + e0 * (min0 + max0) * 0.5
                + e1 * (min1 + max1) * 0.5
                + e2 * (min2 + max2) * 0.5;
            double hx = (max0 - min0) * 0.5;
            double hy = (max1 - min1) * 0.5;
            double hz = (max2 - min2) * 0.5;
            var pl = new Plane(c, e0, e1);
            return new Box(pl, new Interval(-hx, hx), new Interval(-hy, hy), new Interval(-hz, hz));
        }

        // ── sampling ───────────────────────────────────────────────────────

        static bool TrySamplePoints(Brep brep, out List<Point3d> points, out string message)
        {
            points = null;
            message = null;
            try
            {
                if (brep == null || !brep.IsValid)
                {
                    message = "Invalid Brep.";
                    return false;
                }
                if (brep.Faces.Count == 0)
                {
                    message = "Brep has no faces.";
                    return false;
                }

                int nFaces = brep.Faces.Count;
                var faceAreas = new double[nFaces];
                double totalArea = 0;
                Mesh[] faceMeshes = Mesh.CreateFromBrep(brep, MeshingParameters.FastRenderMesh);
                bool meshesOk = faceMeshes != null && faceMeshes.Length == nFaces;

                for (int fi = 0; fi < nFaces; fi++)
                {
                    double a = FaceAreaForSamplingQuota(brep.Faces[fi], meshesOk ? faceMeshes[fi] : null);
                    if (a < 0 || double.IsNaN(a) || double.IsInfinity(a))
                        a = 0;
                    faceAreas[fi] = a;
                    totalArea += a;
                }

                int[] quotas = AllocateFaceSampleQuotas(faceAreas, totalArea, TargetSurfaceSampleCount);
                points = new List<Point3d>(TargetSurfaceSampleCount + 16);

                var rnd = new Random();

                for (int fi = 0; fi < nFaces; fi++)
                {
                    int quota = quotas[fi];
                    if (quota <= 0)
                        continue;

                    BrepFace face = brep.Faces[fi];
                    int countBeforeFace = points.Count;

                    if (meshesOk && faceMeshes[fi] != null && faceMeshes[fi].Faces.Count > 0)
                        AppendRandomPointsOnFaceMesh(faceMeshes[fi], quota, rnd, points);

                    int shortfall = quota - (points.Count - countBeforeFace);
                    if (shortfall > 0)
                        AddRandomUvSurfacePoints(face, shortfall, rnd, points);

                    shortfall = quota - (points.Count - countBeforeFace);
                    if (shortfall > 0)
                        AppendFaceBoundarySamples(face, brep, points, countBeforeFace + quota);
                }

                if (points.Count < 4)
                {
                    for (int fi = 0; fi < nFaces && points.Count < 4; fi++)
                        AppendFaceBoundarySamples(brep.Faces[fi], brep, points, 4);
                }

                if (points.Count < 4)
                {
                    message = "Too few sample points after surface sampling.";
                    points = null;
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                points = null;
                return false;
            }
        }

        /// <summary>Face area for quota: tessellated mesh sum if available, otherwise <see cref="AreaMassProperties"/>.</summary>
        static double FaceAreaForSamplingQuota(BrepFace face, Mesh tessellation)
        {
            if (face == null)
                return 0;

            if (tessellation != null && tessellation.Faces.Count > 0)
            {
                Mesh work = tessellation.DuplicateMesh();
                work.Faces.ConvertQuadsToTriangles();
                double s = 0;
                for (int i = 0; i < work.Faces.Count; i++)
                {
                    MeshFace f = work.Faces[i];
                    Point3d a = work.Vertices[f.A];
                    Point3d b = work.Vertices[f.B];
                    Point3d c = work.Vertices[f.C];
                    double ar = 0.5 * Vector3d.CrossProduct(b - a, c - a).Length;
                    if (ar > 0 && !double.IsNaN(ar) && !double.IsInfinity(ar))
                        s += ar;
                }
                if (s > 1e-18)
                    return s;
            }

            AreaMassProperties amp = AreaMassProperties.Compute(face);
            if (amp != null && amp.Area > 0 && !double.IsNaN(amp.Area) && !double.IsInfinity(amp.Area))
                return amp.Area;
            return 0;
        }

        /// <summary>
        /// Splits <paramref name="targetTotal"/> samples across faces using face area / total area (largest remainder).
        /// If total area is zero, samples are split evenly across faces.
        /// </summary>
        static int[] AllocateFaceSampleQuotas(double[] faceAreas, double totalArea, int targetTotal)
        {
            int n = faceAreas.Length;
            var q = new int[n];
            if (n == 0 || targetTotal <= 0)
                return q;

            if (totalArea <= Rhino.RhinoMath.ZeroTolerance)
            {
                int baseQ = targetTotal / n;
                int rem = targetTotal % n;
                for (int i = 0; i < n; i++)
                    q[i] = baseQ + (i < rem ? 1 : 0);
                return q;
            }

            int assigned = 0;
            var remainders = new (int idx, double frac)[n];
            for (int i = 0; i < n; i++)
            {
                double exact = targetTotal * (faceAreas[i] / totalArea);
                int f = (int)Math.Floor(exact);
                if (f < 0) f = 0;
                q[i] = f;
                assigned += f;
                remainders[i] = (i, exact - f);
            }

            int deficit = targetTotal - assigned;
            if (deficit > 0)
            {
                Array.Sort(remainders, (a, b) => b.frac.CompareTo(a.frac));
                for (int k = 0; k < deficit && k < n; k++)
                    q[remainders[k].idx]++;
            }

            return q;
        }

        /// <summary>Uniform random points on mesh triangles, weighted by triangle area (after quads → triangles).</summary>
        static void AppendRandomPointsOnFaceMesh(Mesh mesh, int count, Random rnd, List<Point3d> dest)
        {
            if (mesh == null || count <= 0)
                return;

            Mesh work = mesh.DuplicateMesh();
            work.Faces.ConvertQuadsToTriangles();
            int nF = work.Faces.Count;
            if (nF == 0)
                return;

            var cum = new double[nF];
            var ax = new Point3d[nF];
            var bx = new Point3d[nF];
            var cx = new Point3d[nF];
            double sum = 0;
            for (int i = 0; i < nF; i++)
            {
                MeshFace f = work.Faces[i];
                Point3d a = work.Vertices[f.A];
                Point3d b = work.Vertices[f.B];
                Point3d c = work.Vertices[f.C];
                ax[i] = a;
                bx[i] = b;
                cx[i] = c;
                double ar = 0.5 * Vector3d.CrossProduct(b - a, c - a).Length;
                if (ar < 1e-18 || double.IsNaN(ar) || double.IsInfinity(ar))
                    ar = 0;
                sum += ar;
                cum[i] = sum;
            }

            if (sum <= 1e-18)
                return;

            for (int k = 0; k < count; k++)
            {
                double t = rnd.NextDouble() * sum;
                int idx = 0;
                while (idx < nF - 1 && cum[idx] < t)
                    idx++;

                Point3d a = ax[idx], b = bx[idx], c = cx[idx];
                double r1 = rnd.NextDouble();
                double r2 = rnd.NextDouble();
                double sr1 = Math.Sqrt(r1);
                Point3d p = a * (1.0 - sr1) + b * (sr1 * (1.0 - r2)) + c * (sr1 * r2);
                dest.Add(p);
            }
        }

        /// <summary>Random (u,v) in face domains; keeps points with <see cref="PointFaceRelation"/> not Exterior.</summary>
        static void AddRandomUvSurfacePoints(BrepFace face, int count, Random rnd, List<Point3d> dest)
        {
            if (count <= 0)
                return;

            Interval uDom = face.Domain(0);
            Interval vDom = face.Domain(1);
            int maxAttempts = Math.Max(count * 100, 400);
            int added = 0;
            for (int attempts = 0; added < count && attempts < maxAttempts; attempts++)
            {
                double u = uDom.ParameterAt(rnd.NextDouble());
                double v = vDom.ParameterAt(rnd.NextDouble());
                if (face.IsPointOnFace(u, v) == PointFaceRelation.Exterior)
                    continue;
                Point3d p = face.PointAt(u, v);
                if (p.IsValid)
                {
                    dest.Add(p);
                    added++;
                }
            }
        }

        /// <summary>Subdivide boundary edges of this face until <paramref name="minTotal"/> points are collected.</summary>
        static void AppendFaceBoundarySamples(BrepFace face, Brep brep, List<Point3d> into, int minTotal)
        {
            if (into.Count >= minTotal)
                return;

            foreach (BrepLoop loop in face.Loops)
            {
                foreach (BrepTrim trim in loop.Trims)
                {
                    if (into.Count >= minTotal)
                        return;
                    if (trim.TrimType == BrepTrimType.Singular)
                        continue;

                    BrepEdge edge = trim.Edge;
                    if (edge == null)
                        continue;
                    Curve crv = edge.EdgeCurve;
                    if (crv == null)
                        continue;

                    int need = minTotal - into.Count;
                    int n = Math.Max(4, Math.Min(64, need + 4));
                    Interval d = edge.Domain;
                    for (int k = 0; k < n; k++)
                    {
                        double t = n <= 1 ? d.Mid : d.ParameterAt(k / (double)(n - 1));
                        into.Add(crv.PointAt(t));
                        if (into.Count >= minTotal)
                            return;
                    }
                }
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

        /// <summary>Second and third axes: dominant direction in the plane orthogonal to <paramref name="e0"/> (covariance iteration).</summary>
        static bool TrySecondPrincipalDirection(double[,] c, Vector3d e0, out Vector3d e1, out Vector3d e2)
        {
            e1 = Vector3d.Zero;
            e2 = Vector3d.Zero;
            Vector3d v = Math.Abs(e0 * Vector3d.XAxis) < 0.9 ? Vector3d.XAxis : Vector3d.YAxis;
            double dot = v * e0;
            v = new Vector3d(v.X - dot * e0.X, v.Y - dot * e0.Y, v.Z - dot * e0.Z);
            if (v.Length < 1e-12)
            {
                v = Vector3d.ZAxis;
                dot = v * e0;
                v = new Vector3d(v.X - dot * e0.X, v.Y - dot * e0.Y, v.Z - dot * e0.Z);
            }
            if (v.Length < 1e-12 || !v.Unitize())
                return false;

            for (int it = 0; it < 40; it++)
            {
                double nx = c[0, 0] * v.X + c[0, 1] * v.Y + c[0, 2] * v.Z;
                double ny = c[1, 0] * v.X + c[1, 1] * v.Y + c[1, 2] * v.Z;
                double nz = c[2, 0] * v.X + c[2, 1] * v.Y + c[2, 2] * v.Z;
                dot = nx * e0.X + ny * e0.Y + nz * e0.Z;
                nx -= dot * e0.X;
                ny -= dot * e0.Y;
                nz -= dot * e0.Z;
                double len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                if (len < 1e-18)
                    return false;
                v = new Vector3d(nx / len, ny / len, nz / len);
            }

            e1 = v;
            e2 = Vector3d.CrossProduct(e0, e1);
            return e2.Unitize();
        }

        static bool TryUnitizeOrthonormal(ref Vector3d u, ref Vector3d v, ref Vector3d w)
        {
            if (!u.Unitize())
                return false;
            v -= u * (u * v);
            if (!v.Unitize())
                return false;
            w = Vector3d.CrossProduct(u, v);
            return w.Unitize();
        }

        static void ProjectExtents(List<Point3d> pts, Point3d centroid,
            Vector3d e0, Vector3d e1, Vector3d e2,
            out double min0, out double max0, out double min1, out double max1, out double min2, out double max2)
        {
            double minA = double.MaxValue, maxA = double.MinValue;
            double minB = double.MaxValue, maxB = double.MinValue;
            double minC = double.MaxValue, maxC = double.MinValue;
            foreach (var p in pts)
            {
                Vector3d r = p - centroid;
                double s0 = r * e0;
                double s1 = r * e1;
                double s2 = r * e2;
                if (s0 < minA) minA = s0;
                if (s0 > maxA) maxA = s0;
                if (s1 < minB) minB = s1;
                if (s1 > maxB) maxB = s1;
                if (s2 < minC) minC = s2;
                if (s2 > maxC) maxC = s2;
            }
            min0 = minA;
            max0 = maxA;
            min1 = minB;
            max1 = maxB;
            min2 = minC;
            max2 = maxC;
        }

        static void AxisExtents(List<Point3d> pts, Point3d centroid,
            Vector3d e0, Vector3d e1, Vector3d e2,
            out double L, out double w, out double h, out Line axisLine, out Box axisAlignedBox)
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
            axisAlignedBox = BuildBoxFromObbExtents(centroid, e0, e1, e2, min0, max0, min1, max1, min2, max2);
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
