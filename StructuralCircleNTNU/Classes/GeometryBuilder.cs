using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Generates Rhino display geometry for Element types positioned at an origin point.
    /// Priority: user-supplied GeometryBrep → built solid → axis line fallback.
    /// </summary>
    public static class GeometryBuilder
    {
        public static GeometryBase BuildElementGeometry(Element element, Point3d origin)
        {
            var move = Transform.Translation(new Vector3d(origin));

            if (element.GeometryBrep != null)
            {
                var dup = element.GeometryBrep.DuplicateBrep();
                dup.Transform(move);
                return dup;
            }

            if (element is Beam beam)
                return BuildBeamGeometry(beam, move);

            if (element is Plate plate)
                return BuildPlateGeometry(plate, move);

            if (element.AxisLine.IsValid)
            {
                var ln = element.AxisLine;
                ln.Transform(move);
                return new LineCurve(ln);
            }

            return null;
        }

        static GeometryBase BuildBeamGeometry(Beam beam, Transform xform)
        {
            if (beam.Section != null && beam.AxisLine.IsValid && beam.Length > 0)
            {
                double w = beam.Section.Width;
                double h = beam.Section.Height;
                double l = beam.Length;

                var box = new Box(Plane.WorldXY,
                    new Interval(0, l),
                    new Interval(-w / 2.0, w / 2.0),
                    new Interval(0, h));

                var brep = box.ToBrep();
                if (brep != null && brep.IsValid)
                {
                    brep.Transform(xform);
                    return brep;
                }
            }

            if (beam.AxisLine.IsValid)
            {
                var ln = beam.AxisLine;
                ln.Transform(xform);
                return new LineCurve(ln);
            }

            return null;
        }

        static GeometryBase BuildPlateGeometry(Plate plate, Transform xform)
        {
            double length    = plate.Length;
            double thickness = plate.Section?.Thickness ?? 0;
            double panelW    = plate.Section?.Width ?? 0;

            if (length > 0 && thickness > 0 && panelW > 0)
            {
                var box = new Box(Plane.WorldXY,
                    new Interval(0, length),
                    new Interval(-panelW / 2.0, panelW / 2.0),
                    new Interval(-thickness / 2.0, thickness / 2.0));

                var brep = box.ToBrep();
                if (brep != null && brep.IsValid)
                {
                    brep.Transform(xform);
                    return brep;
                }
            }

            if (plate.AxisSurface != null)
            {
                var edge = plate.AxisSurface.ToBrep();
                if (edge != null)
                {
                    edge.Transform(xform);
                    return edge;
                }
            }

            if (plate.AxisLine.IsValid)
            {
                var ln = plate.AxisLine;
                ln.Transform(xform);
                return new LineCurve(ln);
            }

            return null;
        }

        /// <summary>Midpoint of the element axis, offset from origin — used for label placement.</summary>
        public static Point3d LabelPoint(Element element, Point3d origin)
        {
            double half = 0;
            if (element is Beam b && b.Length > 0)      half = b.Length / 2.0;
            else if (element is Plate p && p.Length > 0) half = p.Length / 2.0;
            return new Point3d(origin.X + half, origin.Y, origin.Z);
        }
    }
}
