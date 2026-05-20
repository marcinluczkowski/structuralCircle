using System;
using System.Collections.Generic;
#pragma warning disable CS0618 // obsolete ReleaseVersion.IFC4
using GeometryGym.Ifc;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Exports arbitrary Rhino Breps as structural IFC elements or as <see cref="IfcMechanicalFastener"/>
    /// with faceted B-rep geometry.
    /// Structural: Beam, Column, Wall, Slab, Member.
    /// Fasteners use <see cref="IfcMechanicalFastener"/> with the requested predefined type.
    /// </summary>
    public static class IfcBrepExporter
    {
        public enum StructuralType { Beam, Column, Wall, Slab, Member }

        /// <summary>Human-readable IFC types accepted by extended Brep export (structural + common fasteners).</summary>
        public static readonly string ExtendedTypeHelp =
            "Structural: Beam, Column, Wall, Slab, Member. " +
            "Mechanical fastener (IfcMechanicalFastener): Bolt, Screw, Nail, Dowel, Rivet, AnchorBolt, NailPlate";

        /// <summary>
        /// Parses <paramref name="text"/> into either a structural IFC type or a mechanical fastener predefined type.
        /// Fastener names are matched case-insensitively after removing spaces, hyphens and underscores
        /// (e.g. &quot;Anchor bolt&quot;, &quot;NAIL PLATE&quot;).
        /// </summary>
        public static bool TryResolveExportElementType(string text,
            out StructuralType structuralType,
            out bool isStructural,
            out IfcMechanicalFastenerTypeEnum fastenerType,
            out bool isFastener)
        {
            structuralType = default;
            isStructural = false;
            fastenerType = default;
            isFastener = false;

            if (string.IsNullOrWhiteSpace(text))
                return false;

            if (Enum.TryParse(text.Trim(), true, out StructuralType st))
            {
                structuralType = st;
                isStructural = true;
                return true;
            }

            var norm = NonAlphaNumericSpaces(text.Trim());
            if (!Enum.TryParse(norm, true, out IfcMechanicalFastenerTypeEnum ft))
                return false;
            if (ft == IfcMechanicalFastenerTypeEnum.NOTDEFINED || ft == IfcMechanicalFastenerTypeEnum.USERDEFINED)
                return false;

            fastenerType = ft;
            isFastener = true;
            return true;
        }

        static string NonAlphaNumericSpaces(string s)
        {
            var chars = new System.Text.StringBuilder(s.Length);
            foreach (char c in s)
            {
                if (char.IsLetterOrDigit(c))
                    chars.Append(char.ToUpperInvariant(c));
                // drop spaces / punctuation so "Anchor bolt" → "ANCHORBOLT"
            }
            return chars.ToString();
        }

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
            string typeLabel = elementType.ToString();
            return ExportCore(breps, names, materialName, filePath, schemaVersion, projectName, buildingName,
                typeLabel,
                (db, storey, placement, shape, eltName)
                    => CreateIfcElement(db, storey, elementType, placement, shape, eltName));
        }

        /// <summary>
        /// Export Breps as <see cref="IfcMechanicalFastener"/> with the given IFC predefined type and faceted geometry.
        /// </summary>
        public static string ExportMechanicalFasteners(
            List<Brep> breps,
            List<string> names,
            IfcMechanicalFastenerTypeEnum predefinedType,
            string materialName,
            string filePath,
            string schemaVersion = "IFC4",
            string projectName   = "Project",
            string buildingName  = "Building")
        {
            string typeLabel = $"IfcMechanicalFastener.{predefinedType}";
            return ExportCore(breps, names, materialName, filePath, schemaVersion, projectName, buildingName,
                typeLabel,
                (db, storey, placement, shape, eltName) =>
                {
                    var fastener = new IfcMechanicalFastener(storey, placement, shape);
                    fastener.Name = eltName;
                    fastener.PredefinedType = predefinedType;
                    return fastener;
                });
        }

        static string ExportCore(
            List<Brep> breps,
            List<string> names,
            string materialName,
            string filePath,
            string schemaVersion,
            string projectName,
            string buildingName,
            string typeReportLabel,
            Func<DatabaseIfc, IfcBuildingStorey, IfcLocalPlacement, IfcProductDefinitionShape, string, IfcElement> createElement)
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

                string eltName = (names != null && i < names.Count && !string.IsNullOrEmpty(names[i]))
                    ? names[i]
                    : $"{StripIfcPrefixForDefaultName(typeReportLabel)}_{i + 1}";

                try
                {
                    var facetedBrep = BrepToFacetedBrep(db, brep);
                    var shape       = new IfcShapeRepresentation(facetedBrep);
                    var prodShape   = new IfcProductDefinitionShape(shape);
                    var placement   = new IfcLocalPlacement(
                        new IfcAxis2Placement3D(new IfcCartesianPoint(db, 0, 0, 0)));

                    var ifcElem = createElement(db, storey, placement, prodShape, eltName);
                    new IfcRelAssociatesMaterial(mat, new List<IfcDefinitionSelect> { ifcElem });
                    nOk++;
                }
                catch (Exception ex)
                {
                    nErr++;
                    report.AppendLine($"  Brep {i} [{eltName}]: ERROR – {ex.Message}");
                }
            }

            db.WriteFile(filePath);

            report.Insert(0, $"IFC Export ({schemaVersion})\n" +
                             $"  Type: {typeReportLabel}  OK: {nOk}  Errors: {nErr}\n" +
                             $"  File: {filePath}\n\n");
            return report.ToString();
        }

        static string StripIfcPrefixForDefaultName(string typeReportLabel)
        {
            const string prefix = "IfcMechanicalFastener.";
            if (typeReportLabel.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return typeReportLabel.Substring(prefix.Length);
            return typeReportLabel;
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
