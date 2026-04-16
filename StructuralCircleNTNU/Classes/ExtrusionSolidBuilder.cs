using System;
using Rhino;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Builds closed Breps by extruding a rectangular cross-section profile along +X.
    /// Uses Extrusion.Create for capped solids; falls back to Box when extrusion fails.
    /// </summary>
    public static class ExtrusionSolidBuilder
    {
        public static double DefaultTolerance => Math.Max(RhinoMath.ZeroTolerance, 1e-4);

        public static void AssignExtrudedBrep(Element element, double? tolerance = null)
        {
            double tol = tolerance ?? DefaultTolerance;

            if (element is Beam beam)
            {
                if (TryCreateBeamExtrusion(beam, tol, out Brep brep))
                    beam.GeometryBrep = brep;
            }
            else if (element is Plate plate)
            {
                if (TryCreatePlateExtrusion(plate, tol, out Brep brep))
                    plate.GeometryBrep = brep;
            }
        }

        public static bool TryCreateBeamExtrusion(Beam beam, double tol, out Brep brep)
        {
            brep = null;
            if (beam?.Section == null || !beam.AxisLine.IsValid || beam.Length <= tol)
                return false;

            brep = ExtrudeRectProfileAlongX(beam.Section.Width, beam.Section.Height, beam.Length, tol, centreZ: false);
            return brep != null;
        }

        public static bool TryCreatePlateExtrusion(Plate plate, double tol, out Brep brep)
        {
            brep = null;
            if (plate?.Section == null || plate.Length <= tol) return false;

            double t = plate.Section.Thickness;
            double w = plate.Section.Width;
            if (t <= tol || w <= tol) return false;

            brep = ExtrudeRectProfileAlongX(w, t, plate.Length, tol, centreZ: true);
            return brep != null;
        }

        // ── internals ────────────────────────────────────────────────────

        static Brep ExtrudeRectProfileAlongX(double widthY, double heightZ, double lengthX,
            double tol, bool centreZ)
        {
            Interval yi = new Interval(-widthY / 2.0, widthY / 2.0);
            Interval zi = centreZ
                ? new Interval(-heightZ / 2.0, heightZ / 2.0)
                : new Interval(0, heightZ);

            var rect = new Rectangle3d(Plane.WorldYZ, yi, zi);
            Curve profile = rect.ToNurbsCurve();

            if (profile == null || !profile.IsClosed)
                return FallbackBox(widthY, heightZ, lengthX, centreZ);

            Brep brep = null;
            try
            {
                var extrusion = Extrusion.Create(profile, lengthX, true);
                if (extrusion != null)
                    brep = extrusion.ToBrep();
            }
            catch { brep = null; }

            if (brep == null || !brep.IsValid)
                brep = FallbackBox(widthY, heightZ, lengthX, centreZ);

            return FinalizeClosedSolid(brep, tol);
        }

        static Brep FallbackBox(double w, double h, double L, bool centreZ)
        {
            Interval zi = centreZ ? new Interval(-h / 2.0, h / 2.0) : new Interval(0, h);
            var box = new Box(Plane.WorldXY, new Interval(0, L), new Interval(-w / 2.0, w / 2.0), zi);
            return box.ToBrep();
        }

        static Brep FinalizeClosedSolid(Brep brep, double tol)
        {
            if (brep == null || !brep.IsValid) return null;

            if (!brep.IsSolid)
            {
                Brep capped = brep.CapPlanarHoles(tol);
                if (capped != null && capped.IsValid)
                {
                    brep.Dispose();
                    brep = capped;
                }
            }

            if (!brep.IsSolid) return null;

            try { brep.MergeCoplanarFaces(tol); } catch { }
            return brep;
        }
    }
}
