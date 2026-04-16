using System;
using System.Collections.Generic;
#pragma warning disable CS0618 // obsolete ReleaseVersion.IFC4
using GeometryGym.Ifc;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Exports StructuralCircleNTNU Beam / Plate elements to an IFC file.
    /// Geometry is built from section dimensions and represented as IfcFacetedBrep.
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

            // Spatial hierarchy: Site → Project with units → Building → Storey
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
            double w = beam.Section?.Width  > 0 ? beam.Section.Width  : 0.1;
            double h = beam.Section?.Height > 0 ? beam.Section.Height : 0.2;
            double L = beam.Length          > 0 ? beam.Length         : 1.0;

            var axis = beam.AxisLine.IsValid ? beam.AxisLine
                                             : new Line(Point3d.Origin, new Point3d(L, 0, 0));
            var brep = BoxFromBeamSection(w, h, L, axis);

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
            double t = plate.Section?.Thickness > 0 ? plate.Section.Thickness : 0.1;
            double w = plate.Section?.Width     > 0 ? plate.Section.Width     : 1.0;
            double L = plate.Length             > 0 ? plate.Length            : 1.0;

            var axis = plate.AxisLine.IsValid ? plate.AxisLine
                                              : new Line(Point3d.Origin, new Point3d(L, 0, 0));
            var brep = BoxFromPlateSection(w, t, L, axis);

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

        // ── geometry helpers ──────────────────────────────────────────────

        static Brep BoxFromBeamSection(double w, double h, double L, Line axis)
        {
            var box = new Box(Plane.WorldXY,
                new Interval(0, L), new Interval(-w / 2.0, w / 2.0), new Interval(0, h));
            var brep = box.ToBrep();
            brep.Transform(TransformFromAxis(axis));
            return brep;
        }

        static Brep BoxFromPlateSection(double w, double t, double L, Line axis)
        {
            var box = new Box(Plane.WorldXY,
                new Interval(0, L), new Interval(-w / 2.0, w / 2.0), new Interval(-t / 2.0, t / 2.0));
            var brep = box.ToBrep();
            brep.Transform(TransformFromAxis(axis));
            return brep;
        }

        static Transform TransformFromAxis(Line axis)
        {
            var dir = axis.Direction; dir.Unitize();
            return Transform.PlaneToPlane(Plane.WorldXY, new Plane(axis.From, dir));
        }

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
