using System;
using System.Collections.Generic;
using Rhino.Geometry;
using Rhino.Geometry.Intersect;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>Cross-section dimensions from Brep: plane cuts at 25/50/75% along PCA axis and min-volume oriented box.</summary>
    public sealed class BrepSectionReadResult
    {
        public Line PcaAxis { get; set; }
        /// <summary>Member length along PCA axis (m).</summary>
        public double LengthAlongAxis { get; set; }
        public double WidthFromPlaneCuts { get; set; }
        public double HeightFromPlaneCuts { get; set; }
        public bool PlaneCutsOk { get; set; }
        public double LengthFromMinVolumeBox { get; set; }
        public double WidthFromMinVolumeBox { get; set; }
        public double HeightFromMinVolumeBox { get; set; }
        public bool MinVolumeBoxOk { get; set; }
        public string Notes { get; set; }
    }

    public static class BrepSectionFromBrep
    {
        static readonly double[] PlaneTParams = { 0.25, 0.5, 0.75 };

        public static bool TryRead(Brep brep, double tolerance, out BrepSectionReadResult result, out string error)
        {
            result = new BrepSectionReadResult();
            error = null;

            if (brep == null || !brep.IsValid)
            {
                error = "Invalid Brep.";
                return false;
            }

            if (!BrepElementBuilder.TryGetPcaAxisAndFrame(brep, out Line axis, out Vector3d e0, out Vector3d e1, out Vector3d e2,
                    out List<Point3d> samples, out string pcaMsg))
            {
                error = pcaMsg ?? "PCA failed.";
                return false;
            }

            result.PcaAxis = axis;
            result.LengthAlongAxis = axis.Length;

            var notes = new List<string>();
            var widths = new List<double>();
            var heights = new List<double>();

            Brep brepDup = brep.DuplicateBrep();
            double tol = Math.Max(tolerance, 1e-9);

            foreach (double t in PlaneTParams)
            {
                Point3d p = axis.PointAt(t);
                var pl = new Plane(p, e1, e2);
                if (!Intersection.BrepPlane(brepDup, pl, tol, out Curve[] crvs, out _))
                {
                    notes.Add($"t={t}: no intersection.");
                    continue;
                }
                if (crvs == null || crvs.Length == 0)
                {
                    notes.Add($"t={t}: empty curves.");
                    continue;
                }

                if (!TrySectionRectangleDims(crvs, tol, pl.Origin, e1, e2, out double w, out double h, out string cutNote))
                {
                    notes.Add($"t={t}: {cutNote}");
                    continue;
                }

                widths.Add(w);
                heights.Add(h);
            }

            if (widths.Count > 0)
            {
                FilterAndAverage(widths, heights, 0.5, out double aw, out double ah);
                result.WidthFromPlaneCuts = aw;
                result.HeightFromPlaneCuts = ah;
                result.PlaneCutsOk = true;
            }
            else
            {
                result.PlaneCutsOk = false;
                notes.Add("All plane cuts failed.");
            }

            if (BrepElementBuilder.TryMinimumVolumeOrientedBoxFromPoints(samples, out Box box, out string boxMsg) && box.IsValid)
            {
                MapMinVolumeBoxToLengthWidthHeight(box, e0, out double lb, out double wb, out double hb);
                result.LengthFromMinVolumeBox = lb;
                result.WidthFromMinVolumeBox = wb;
                result.HeightFromMinVolumeBox = hb;
                result.MinVolumeBoxOk = true;
            }
            else
            {
                result.MinVolumeBoxOk = false;
                notes.Add(boxMsg ?? "Min-volume box failed.");
            }

            result.Notes = notes.Count > 0 ? string.Join(" ", notes) : null;
            if (!result.PlaneCutsOk && !result.MinVolumeBoxOk)
            {
                error = "Plane cuts and min-volume box both failed.";
                return false;
            }

            return true;
        }

        /// <summary>Assigns box edge parallel to member axis as length; other two edges as width (larger) and height.</summary>
        static void MapMinVolumeBoxToLengthWidthHeight(Box box, Vector3d memberAxis, out double length, out double width, out double height)
        {
            memberAxis.Unitize();
            double lx = box.X.Length;
            double ly = box.Y.Length;
            double lz = box.Z.Length;
            Plane bp = box.Plane;
            var ux = new Vector3d(bp.XAxis);
            var uy = new Vector3d(bp.YAxis);
            var uz = new Vector3d(bp.ZAxis);
            ux.Unitize();
            uy.Unitize();
            uz.Unitize();

            double px = lx * Math.Abs(memberAxis * ux);
            double py = ly * Math.Abs(memberAxis * uy);
            double pz = lz * Math.Abs(memberAxis * uz);

            double lenEdge, crossA, crossB;
            if (px >= py && px >= pz)
            {
                lenEdge = lx;
                crossA = ly;
                crossB = lz;
            }
            else if (py >= px && py >= pz)
            {
                lenEdge = ly;
                crossA = lx;
                crossB = lz;
            }
            else
            {
                lenEdge = lz;
                crossA = lx;
                crossB = ly;
            }

            length = lenEdge;
            width = Math.Max(crossA, crossB);
            height = Math.Min(crossA, crossB);
        }

        /// <summary>Drop samples whose w or h differs by more than <paramref name="relativeTolerance"/> from the mean of the other samples; average the rest.</summary>
        static void FilterAndAverage(List<double> widths, List<double> heights, double relativeTolerance,
            out double avgW, out double avgH)
        {
            avgW = avgH = 0;
            int n = widths.Count;
            if (n == 0)
                return;
            if (heights.Count != n)
                throw new ArgumentException("Width and height lists must match.");

            if (n == 1)
            {
                avgW = widths[0];
                avgH = heights[0];
                return;
            }

            var keep = new List<int>();
            for (int i = 0; i < n; i++)
            {
                double sumW = 0, sumH = 0;
                int cnt = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                        continue;
                    sumW += widths[j];
                    sumH += heights[j];
                    cnt++;
                }

                double mw = cnt > 0 ? sumW / cnt : widths[i];
                double mh = cnt > 0 ? sumH / cnt : heights[i];
                bool badW = mw > Rhino.RhinoMath.ZeroTolerance && Math.Abs(widths[i] - mw) / mw > relativeTolerance;
                bool badH = mh > Rhino.RhinoMath.ZeroTolerance && Math.Abs(heights[i] - mh) / mh > relativeTolerance;
                if (!badW && !badH)
                    keep.Add(i);
            }

            if (keep.Count == 0)
            {
                for (int i = 0; i < n; i++)
                {
                    avgW += widths[i];
                    avgH += heights[i];
                }
                avgW /= n;
                avgH /= n;
                return;
            }

            avgW = 0;
            avgH = 0;
            foreach (int i in keep)
            {
                avgW += widths[i];
                avgH += heights[i];
            }
            avgW /= keep.Count;
            avgH /= keep.Count;
        }

        static bool TrySectionRectangleDims(Curve[] segments, double tolerance, Point3d planeOrigin, Vector3d e1, Vector3d e2,
            out double width, out double height, out string note)
        {
            width = height = 0;
            note = null;

            Curve[] joined = Curve.JoinCurves(segments, tolerance);
            var candidates = new List<Curve>();
            if (joined != null && joined.Length > 0)
            {
                foreach (var c in joined)
                {
                    if (c != null && c.IsValid)
                        candidates.Add(c);
                }
            }
            else
            {
                foreach (var c in segments)
                {
                    if (c != null && c.IsValid)
                        candidates.Add(c);
                }
            }

            Curve best = null;
            double bestArea = 0;
            foreach (var c in candidates)
            {
                if (!c.IsClosed)
                    continue;
                var amp = AreaMassProperties.Compute(c);
                if (amp == null)
                    continue;
                if (amp.Area > bestArea)
                {
                    bestArea = amp.Area;
                    best = c;
                }
            }

            if (best == null)
            {
                double bestScore = 0;
                foreach (var c in candidates)
                {
                    if (TryCurveSpanInFrame(c, planeOrigin, e1, e2, out double span1, out double span2))
                    {
                        double score = span1 * span2;
                        if (score > bestScore)
                        {
                            bestScore = score;
                            best = c;
                        }
                    }
                }
            }

            if (best == null)
            {
                note = "no usable section curve.";
                return false;
            }

            if (!TryCurveSpanInFrame(best, planeOrigin, e1, e2, out double su, out double sv))
            {
                note = "could not measure rectangle spans.";
                return false;
            }

            width = Math.Max(su, sv);
            height = Math.Min(su, sv);
            return width > Rhino.RhinoMath.ZeroTolerance;
        }

        static bool TryCurveSpanInFrame(Curve curve, Point3d origin, Vector3d e1, Vector3d e2,
            out double spanU, out double spanV)
        {
            spanU = spanV = 0;
            if (curve == null || !curve.IsValid)
                return false;

            double minU = double.MaxValue, maxU = double.MinValue;
            double minV = double.MaxValue, maxV = double.MinValue;
            bool any = false;

            void consider(Point3d p)
            {
                Vector3d r = p - origin;
                double u = r * e1;
                double v = r * e2;
                if (u < minU) minU = u;
                if (u > maxU) maxU = u;
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
                any = true;
            }

            int steps = 64;
            double len = curve.GetLength();
            if (len > Rhino.RhinoMath.ZeroTolerance)
                steps = Math.Min(256, Math.Max(32, (int)(len / Math.Max(Rhino.RhinoMath.ZeroTolerance * 100, 1e-6))));

            for (int i = 0; i <= steps; i++)
            {
                double tn = i / (double)steps;
                consider(curve.PointAtNormalizedLength(tn));
            }

            if (!any)
                return false;

            spanU = maxU - minU;
            spanV = maxV - minV;
            return spanU >= 0 && spanV >= 0;
        }
    }
}
