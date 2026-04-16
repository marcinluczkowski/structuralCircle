using System;
using System.Collections.Generic;
#pragma warning disable CS0618 // obsolete ReleaseVersion.IFC4
using GeometryGym.Ifc;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Exports arbitrary Rhino Breps as structural IFC elements (IfcFacetedBrep geometry).
    /// Supported types: Beam, Column, Wall, Slab, Member.
    /// </summary>
    public static class IfcBrepExporter
    {
        public enum StructuralType { Beam, Column, Wall, Slab, Member }

        public static string Export(
            List<Brep> breps,
            List<string> names,
            StructuralType elementType,
            string materialName,
            string filePath,
            string schemaVersion = "IFC4",
            string projectName   = "Project",
            string buildingName  = "Building")
        {
            if (breps == null || breps.Count == 0)
                throw new ArgumentException("No Breps provided.");

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

            var mat = new IfcMaterial(db, string.IsNullOrWhiteSpace(materialName) ? "Unknown" : materialName);

            int nOk = 0, nErr = 0;
            var report = new System.Text.StringBuilder();

            for (int i = 0; i < breps.Count; i++)
            {
                var brep = breps[i];
                if (brep == null || !brep.IsValid)
                {
                    nErr++;
                    report.AppendLine($"  Brep {i}: null or invalid, skipped.");
                    continue;
                }

                string name = (names != null && i < names.Count && !string.IsNullOrEmpty(names[i]))
                    ? names[i]
                    : $"{elementType}_{i + 1}";

                try
                {
                    var facetedBrep = BrepToFacetedBrep(db, brep);
                    var shape       = new IfcShapeRepresentation(facetedBrep);
                    var prodShape   = new IfcProductDefinitionShape(shape);
                    var placement   = new IfcLocalPlacement(
                        new IfcAxis2Placement3D(new IfcCartesianPoint(db, 0, 0, 0)));

                    var ifcElem = CreateIfcElement(db, storey, elementType, placement, prodShape, name);
                    new IfcRelAssociatesMaterial(mat, new List<IfcDefinitionSelect> { ifcElem });
                    nOk++;
                }
                catch (Exception ex)
                {
                    nErr++;
                    report.AppendLine($"  Brep {i} [{name}]: ERROR – {ex.Message}");
                }
            }

            db.WriteFile(filePath);

            report.Insert(0, $"IFC Export ({schemaVersion})\n" +
                             $"  Type: {elementType}  OK: {nOk}  Errors: {nErr}\n" +
                             $"  File: {filePath}\n\n");
            return report.ToString();
        }

        // ── helpers ───────────────────────────────────────────────────────

        static IfcFacetedBrep BrepToFacetedBrep(DatabaseIfc db, Brep brep)
        {
            var mesh   = new Mesh();
            var meshes = Mesh.CreateFromBrep(brep, MeshingParameters.FastRenderMesh);
            if (meshes == null || meshes.Length == 0) throw new Exception("Tessellation produced no mesh.");
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

        static IfcElement CreateIfcElement(DatabaseIfc db, IfcBuildingStorey storey,
            StructuralType type, IfcLocalPlacement placement,
            IfcProductDefinitionShape shape, string name)
        {
            IfcElement elem;

            if (type == StructuralType.Beam)
                elem = new IfcBeam(storey, placement, shape);
            else if (type == StructuralType.Column)
                elem = new IfcColumn(storey, placement, shape);
            else if (type == StructuralType.Wall)
                elem = new IfcWall(storey, placement, shape);
            else if (type == StructuralType.Slab)
            {
                var slab = new IfcSlab(storey, placement, shape);
                slab.PredefinedType = IfcSlabTypeEnum.FLOOR;
                elem = slab;
            }
            else
            {
                // IfcMember (IFC4); fall back to IfcBeam for IFC2x3
                try   { elem = new IfcMember(storey, placement, shape); }
                catch { elem = new IfcBeam(storey, placement, shape);   }
            }

            elem.Name = name;
            return elem;
        }
    }
}
