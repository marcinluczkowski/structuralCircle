using System;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Bank preview: elements in a common layout frame — member along +World Y (axis projected to XY),
    /// cross-section shorter along X, longer along +Z, stacked along Y with stride from bbox.
    /// </summary>
    public static class PreviewBankLayout
    {
        /// <summary>Extent along +Y used for spacing (member length in the preview frame).</summary>
        public static double GetStrideAlongY(Element element)
        {
            if (TryGetPreviewBox(element, out _, out double stride) && stride > 1e-9)
                return stride;
            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
                return element.AxisLine.Length;
            return 0.05;
        }

        /// <summary>Stride along Y and vertical extent along Z for label placement.</summary>
        public static bool TryGetLayoutExtents(Element element, out double strideY, out double depthZ)
        {
            strideY = 0;
            depthZ = 0;
            if (TryGetPreviewBox(element, out Box box, out strideY) && box.IsValid && strideY > 1e-9)
            {
                var bb = box.BoundingBox;
                depthZ = Math.Max(bb.Max.Z - bb.Min.Z, 1e-9);
                return true;
            }

            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
            {
                strideY = element.AxisLine.Length;
                depthZ = 0.05;
                return true;
            }

            return false;
        }

        /// <summary>Label point in world space after geometry is placed at <paramref name="placement"/>.</summary>
        public static Point3d GetLabelPoint(Point3d placement, double lengthY, double dimZ)
        {
            return new Point3d(placement.X, placement.Y + 0.5 * lengthY, placement.Z + 0.5 * dimZ);
        }

        /// <summary>Builds preview geometry in canonical layout (origin at foot; use <see cref="Transform.Translation"/> to place).</summary>
        public static GeometryBase BuildGeometry(Element element)
        {
            if (TryGetPreviewBox(element, out Box box, out _))
            {
                var brep = box.ToBrep();
                return brep?.IsValid == true ? brep : null;
            }

            if (element.AxisLine.IsValid && element.AxisLine.Length > 1e-9)
            {
                var ln = new Line(Point3d.Origin, new Point3d(0, element.AxisLine.Length, 0));
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
                    strideAlongY = L;
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
                    strideAlongY = L;
                    return true;
                }

                if (plate.GeometryBrep != null && plate.GeometryBrep.IsValid)
                    return TrySortedWorldBox(plate.GeometryBrep, out box, out strideAlongY);
            }

            if (element.GeometryBrep != null && element.GeometryBrep.IsValid)
                return TrySortedWorldBox(element.GeometryBrep, out box, out strideAlongY);

            return false;
        }

        /// <summary>WorldXY box: X = shorter cross-section, Y = member length [0,L], Z = longer section [0,long].</summary>
        static Box CanonicalSectionBox(double lengthAlongY, double shortCrossX, double longAlongZ)
        {
            return new Box(Plane.WorldXY,
                new Interval(-0.5 * shortCrossX, 0.5 * shortCrossX),
                new Interval(0, lengthAlongY),
                new Interval(0, longAlongZ));
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
            strideAlongY = L;
            return true;
        }
    }
}
