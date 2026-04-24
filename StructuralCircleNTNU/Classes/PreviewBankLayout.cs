using System;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Bank preview frame (world axes):
    /// Member axis along +Z, shorter cross-section side along ±X, longer cross-section side along ±Y.
    /// Elements are stacked along +World Y in <see cref="Components.Preview.Preview_Bank"/> using
    /// <see cref="GetStrideAlongY"/> (Y-extent of the preview box).
    /// </summary>
    public static class PreviewBankLayout
    {
        /// <summary>Advance along +Y between stacked elements (Y-extent of the preview box).</summary>
        public static double GetStrideAlongY(Element element)
        {
            if (TryGetPreviewBox(element, out Box box, out _) && box.IsValid)
            {
                var bb = box.BoundingBox;
                double dy = bb.Max.Y - bb.Min.Y;
                if (dy > 1e-9) return dy;
            }

            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
                return Math.Max(0.05, element.AxisLine.Length * 0.05);

            return 0.05;
        }

        /// <summary>Stride along stacking Y and member length along +Z (for labels / connectors).</summary>
        public static bool TryGetLayoutExtents(Element element, out double strideAlongY, out double memberLengthZ)
        {
            strideAlongY = 0;
            memberLengthZ = 0;
            if (TryGetPreviewBox(element, out Box box, out _) && box.IsValid)
            {
                var bb = box.BoundingBox;
                strideAlongY = Math.Max(bb.Max.Y - bb.Min.Y, 1e-9);
                memberLengthZ = Math.Max(bb.Max.Z - bb.Min.Z, 1e-9);
                return true;
            }

            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
            {
                strideAlongY = 0.05;
                memberLengthZ = element.AxisLine.Length;
                return true;
            }

            return false;
        }

        /// <summary>Label anchor: centre of member in Z, centre of section in XY at placement origin.</summary>
        public static Point3d GetLabelPoint(Point3d placement, double memberLengthZ)
        {
            return new Point3d(placement.X, placement.Y, placement.Z + 0.5 * memberLengthZ);
        }

        /// <summary>Preview geometry in canonical layout at origin (translate to world with placement).</summary>
        public static GeometryBase BuildGeometry(Element element)
        {
            if (TryGetPreviewBox(element, out Box box, out _))
            {
                var brep = box.ToBrep();
                return brep?.IsValid == true ? brep : null;
            }

            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
            {
                var ln = new Line(Point3d.Origin, new Point3d(0, 0, element.AxisLine.Length));
                return new LineCurve(ln);
            }

            return null;
        }

        static bool TryGetPreviewBox(Element element, out Box box, out double strideAlongY)
        {
            box = Box.Empty;
            strideAlongY = 0;

            if (element is Beam beam)
            {
                if (beam.Section is BeamSection bs
                    && beam.Length > 1e-9 && bs.Width > 1e-9 && bs.Height > 1e-9)
                {
                    double L = beam.Length;
                    double shortDim = Math.Min(bs.Width, bs.Height);
                    double longDim = Math.Max(bs.Width, bs.Height);
                    box = CanonicalSectionBox(L, shortDim, longDim);
                    strideAlongY = longDim;
                    return true;
                }

                if (beam.GeometryBrep != null && beam.GeometryBrep.IsValid)
                    return TrySortedWorldBox(beam.GeometryBrep, out box, out strideAlongY);
            }

            if (element is Plate plate)
            {
                if (plate.Section is PlateSection ps && plate.Length > 1e-9)
                {
                    double L = plate.Length;
                    double t = Math.Max(ps.Thickness, 1e-6);
                    double w = ps.Width > 1e-9 ? ps.Width : Math.Max(L * 0.02, t * 2);
                    double shortDim = Math.Min(w, t);
                    double longDim = Math.Max(w, t);
                    box = CanonicalSectionBox(L, shortDim, longDim);
                    strideAlongY = longDim;
                    return true;
                }

                if (plate.GeometryBrep != null && plate.GeometryBrep.IsValid)
                    return TrySortedWorldBox(plate.GeometryBrep, out box, out strideAlongY);
            }

            if (element.GeometryBrep != null && element.GeometryBrep.IsValid)
                return TrySortedWorldBox(element.GeometryBrep, out box, out strideAlongY);

            return false;
        }

        /// <summary>
        /// WorldXY-based box: plane X = min section side, plane Y = max section side, plane Z = member length (+Z).
        /// </summary>
        static Box CanonicalSectionBox(double memberLengthZ, double shortCrossX, double longCrossY)
        {
            return new Box(Plane.WorldXY,
                new Interval(-0.5 * shortCrossX, 0.5 * shortCrossX),
                new Interval(-0.5 * longCrossY, 0.5 * longCrossY),
                new Interval(0, memberLengthZ));
        }

        static bool TrySortedWorldBox(Brep brep, out Box box, out double strideAlongY)
        {
            box = Box.Empty;
            strideAlongY = 0;
            if (brep == null || !brep.IsValid) return false;

            var bb = brep.GetBoundingBox(true);
            if (!bb.IsValid) return false;

            double dx = bb.Max.X - bb.Min.X;
            double dy = bb.Max.Y - bb.Min.Y;
            double dz = bb.Max.Z - bb.Min.Z;
            double a = Math.Min(Math.Min(dx, dy), dz);
            double c = Math.Max(Math.Max(dx, dy), dz);
            double b = dx + dy + dz - a - c;

            double L = c;
            double shortDim = a;
            double longDim = Math.Max(b, 1e-6);
            box = CanonicalSectionBox(L, shortDim, longDim);
            strideAlongY = longDim;
            return true;
        }
    }
}
