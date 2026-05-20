using System;
using System.Collections.Generic;
#pragma warning disable CS0618 // obsolete ReleaseVersion.IFC4
using GeometryGym.Ifc;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Exports StructuralCircleNTNU Beam / Plate elements to an IFC file.
    ///
    /// Geometry strategy:
    ///   1. If the element has a <c>GeometryBrep</c>, tessellate it in place — the brep already
    ///      carries the correct world position, no transform is applied.
    ///   2. Otherwise build a rectangular box fully centred on the <c>AxisLine</c>: Y = ±w/2,
    ///      Z = ±h/2, so the structural axis passes through the section centre of gravity.
    /// </summary>
    public static class IfcStructuralExporter
    {
        public static string Export(
            List<Element> elements,
            string filePath,
            string schemaVersion = "IFC4",
            string projectName   = "StructuralCircle Project",
            string buildingName  = "Building")
        {
            if (elements == null || elements.Count == 0)
                throw new ArgumentException("No elements provided for IFC export.");

            bool is2x3 = schemaVersion.ToUpperInvariant().Contains("2X3");
            var release = is2x3 ? ReleaseVersion.IFC2x3 : ReleaseVersion.IFC4A2;

            var db = new DatabaseIfc(release);
            db.Factory.ApplicationFullName   = "StructuralCircleNTNU";
            db.Factory.ApplicationIdentifier = "StructuralCircleNTNU";
            db.Factory.ApplicationVersion    = "1.0";

            var site     = new IfcSite(db, "Site");
            var project  = new IfcProject(site, projectName, IfcUnitAssignment.Length.Metre);
            var building = new IfcBuilding(site, buildingName);
            var storey   = new IfcBuildingStorey(building, "Ground Floor", 0.0);

            int nBeams = 0, nSlabs = 0, nErr = 0;
            var report = new System.Text.StringBuilder();

            foreach (var elem in elements)
            {
                try
                {
                    if (elem is Beam beam)
                    {
                        ExportBeam(db, storey, beam);
                        nBeams++;
                    }
                    else if (elem is Plate plate)
                    {
                        ExportSlab(db, storey, plate);
                        nSlabs++;
                    }
                }
                catch (Exception ex)
                {
                    report.AppendLine($"  ERROR [{elem.Id}] {elem.Name}: {ex.Message}");
                    nErr++;
                }
            }

            db.WriteFile(filePath);

            report.Insert(0, $"IFC Export ({schemaVersion})\n" +
                             $"  Beams: {nBeams}  Slabs: {nSlabs}  Errors: {nErr}\n" +
                             $"  File : {filePath}\n\n");
            return report.ToString();
        }

        // ── element builders ──────────────────────────────────────────────

        static void ExportBeam(DatabaseIfc db, IfcBuildingStorey storey, Beam beam)
        {
            var brep = ElementGeometryBrep(beam);

            var shape     = new IfcShapeRepresentation(BrepToFacetedBrep(db, brep));
            var prodShape = new IfcProductDefinitionShape(shape);
            var placement = WorldPlacement(db);

            var ifcBeam = new IfcBeam(storey, placement, prodShape);
            ifcBeam.Name        = beam.Name;
            ifcBeam.Description = $"StructuralCircle Beam [{beam.Id}]";

            var mat = new IfcMaterial(db, beam.Material?.Name ?? "Timber");
            new IfcRelAssociatesMaterial(mat, new List<IfcDefinitionSelect> { ifcBeam });

            new IfcPropertySet(ifcBeam, "StructuralCircle_Properties", BuildProps(db, beam));
        }

        static void ExportSlab(DatabaseIfc db, IfcBuildingStorey storey, Plate plate)
        {
            var brep = ElementGeometryBrep(plate);

            var shape     = new IfcShapeRepresentation(BrepToFacetedBrep(db, brep));
            var prodShape = new IfcProductDefinitionShape(shape);
            var placement = WorldPlacement(db);

            var ifcSlab = new IfcSlab(storey, placement, prodShape);
            ifcSlab.Name           = plate.Name;
            ifcSlab.Description    = $"StructuralCircle Plate [{plate.Id}]";
            ifcSlab.PredefinedType = IfcSlabTypeEnum.FLOOR;

            var mat = new IfcMaterial(db, plate.Material?.Name ?? "Timber");
            new IfcRelAssociatesMaterial(mat, new List<IfcDefinitionSelect> { ifcSlab });

            new IfcPropertySet(ifcSlab, "StructuralCircle_Properties", BuildProps(db, plate));
        }

        // ── geometry ─────────────────────────────────────────────────────

        /// <summary>
        /// Returns the Brep to export.
        /// If the element already carries a valid <c>GeometryBrep</c> it is used as-is (world coordinates).
        /// Otherwise an analytical box is built centred on the axis so the axis = neutral axis.
        /// </summary>
        static Brep ElementGeometryBrep(Element element)
        {
            if (element.GeometryBrep != null && element.GeometryBrep.IsValid)
                return element.GeometryBrep;

            var axis = element.AxisLine.IsValid
                ? element.AxisLine
                : new Line(Point3d.Origin, new Point3d(1, 0, 0));
            double L = axis.Length > 1e-9 ? axis.Length : 1.0;

            double w, h;
            if (element is Beam beam)
            {
                w = beam.Section?.Width  > 0 ? beam.Section.Width  : 0.1;
                h = beam.Section?.Height > 0 ? beam.Section.Height : 0.1;
            }
            else if (element is Plate plate)
            {
                w = plate.Section?.Width     > 0 ? plate.Section.Width     : 1.0;
                h = plate.Section?.Thickness > 0 ? plate.Section.Thickness : 0.1;
            }
            else { w = 0.1; h = 0.1; }

            return CentredBoxOnAxis(axis, w, h, L);
        }

        /// <summary>
        /// Box fully centred on <paramref name="axis"/>: local X = [0, L], Y = [−w/2, +w/2], Z = [−h/2, +h/2].
        /// A frame is built at <c>axis.From</c> with +X along the member so the axis line passes through
        /// the section centre of gravity (the midpoint of both section dimensions).
        /// </summary>
        static Brep CentredBoxOnAxis(Line axis, double w, double h, double L)
        {
            var box = new Box(Plane.WorldXY,
                new Interval(0, L),
                new Interval(-w / 2.0, w / 2.0),
                new Interval(-h / 2.0, h / 2.0));
            var brep = box.ToBrep();

            var dir = axis.Direction;
            if (!dir.Unitize()) return brep;

            // Stable Y: for horizontal/diagonal members keep section Z near world-up.
            // For near-vertical members use world X as the reference.
            Vector3d refUp = Math.Abs(dir * Vector3d.ZAxis) < 0.9
                ? Vector3d.ZAxis
                : Vector3d.XAxis;
            var yAxis = Vector3d.CrossProduct(refUp, dir);
            if (yAxis.Length < 1e-9) return brep;
            yAxis.Unitize();

            // Plane(origin, xAxis, yAxis): xAxis = dir (member), yAxis as computed above.
            brep.Transform(Transform.PlaneToPlane(Plane.WorldXY, new Plane(axis.From, dir, yAxis)));
            return brep;
        }

        // ── IFC helpers ───────────────────────────────────────────────────

        static IfcFacetedBrep BrepToFacetedBrep(DatabaseIfc db, Brep brep)
        {
            var mesh   = new Mesh();
            var meshes = Mesh.CreateFromBrep(brep, MeshingParameters.FastRenderMesh);
            if (meshes == null || meshes.Length == 0) throw new Exception("Tessellation failed.");
            foreach (var m in meshes) mesh.Append(m);
            mesh.Faces.ConvertQuadsToTriangles();

            var faces = new List<IfcFace>();
            foreach (var mf in mesh.Faces)
            {
                var pts = new List<IfcCartesianPoint>
                {
                    new IfcCartesianPoint(db, mesh.Vertices[mf.A].X, mesh.Vertices[mf.A].Y, mesh.Vertices[mf.A].Z),
                    new IfcCartesianPoint(db, mesh.Vertices[mf.B].X, mesh.Vertices[mf.B].Y, mesh.Vertices[mf.B].Z),
                    new IfcCartesianPoint(db, mesh.Vertices[mf.C].X, mesh.Vertices[mf.C].Y, mesh.Vertices[mf.C].Z),
                };
                faces.Add(new IfcFace(new List<IfcFaceBound>
                    { new IfcFaceOuterBound(new IfcPolyLoop(pts), true) }));
            }

            if (faces.Count == 0) throw new Exception("No faces after tessellation.");
            return new IfcFacetedBrep(new IfcClosedShell(faces));
        }

        static IfcLocalPlacement WorldPlacement(DatabaseIfc db)
            => new IfcLocalPlacement(new IfcAxis2Placement3D(new IfcCartesianPoint(db, 0, 0, 0)));

        static List<IfcProperty> BuildProps(DatabaseIfc db, Element element)
        {
            var props = new List<IfcProperty>
            {
                new IfcPropertySingleValue(db, "ElementId",   element.Id),
                new IfcPropertySingleValue(db, "ElementType", element.ElementType),
                new IfcPropertySingleValue(db, "Location",    element.Location ?? ""),
                new IfcPropertySingleValue(db, "Material",    element.Material?.Name ?? ""),
            };

            if (element is Beam b && b.Section != null)
            {
                props.Add(new IfcPropertySingleValue(db, "Width",  b.Section.Width));
                props.Add(new IfcPropertySingleValue(db, "Height", b.Section.Height));
                props.Add(new IfcPropertySingleValue(db, "Length", b.Length));
            }
            else if (element is Plate p && p.Section != null)
            {
                props.Add(new IfcPropertySingleValue(db, "Thickness", p.Section.Thickness));
                props.Add(new IfcPropertySingleValue(db, "Width",     p.Section.Width));
                props.Add(new IfcPropertySingleValue(db, "Length",    p.Length));
            }

            return props;
        }
    }
}
