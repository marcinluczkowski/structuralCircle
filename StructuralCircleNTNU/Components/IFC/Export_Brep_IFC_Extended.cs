using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using GeometryGym.Ifc;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.IFC
{
    /// <summary>
    /// Same as <see cref="Export_Brep_IFC"/>, plus export as <see cref="IfcMechanicalFastener"/>
    /// from Brep geometry (IfcFacetedBrep).
    /// </summary>
    public class Export_Brep_IFC_Extended : GH_Component
    {
        private static readonly string[] StructuralOptions =
            { "Beam", "Column", "Wall", "Slab", "Member" };

        /// <summary>Common steel fasteners (IfcPredefinedType).</summary>
        private static readonly string[] FastenerOptions =
            { "Bolt", "Screw", "Nail", "Dowel", "Rivet", "AnchorBolt", "NailPlate" };

        public Export_Brep_IFC_Extended()
            : base("Brep to IFC (extended)", "BrepIFCx",
                   "Export Rhino Breps to IFC — structural (Beam … Member) or steel fasteners as IfcMechanicalFastener "
                   + "(Bolt, Screw, Nail, Dowel, Rivet, AnchorBolt, NailPlate, plus other IFC predefined types spelling).",
                   "StructuralCircleNTNU", "IFC") { }

        public override Guid ComponentGuid => new Guid("81d4be7f-62a9-4532-9fb1-cc5c4c7d8aa1");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter  ("Breps",         "Breps",     "Geometry to export.",                                  GH_ParamAccess.list);
            pManager.AddTextParameter  ("Names",         "Names",     "Element names (one per Brep). Default: element1, element2, …", GH_ParamAccess.list);
            pManager.AddTextParameter  ("ElementType",   "Type",
                "Structural: Beam | Column | Wall | Slab | Member. Fastener (IfcMechanicalFastener): Bolt | Screw | Nail | "
                + "Dowel | Rivet | AnchorBolt | NailPlate (spaces OK, e.g. 'Anchor bolt'). Other IFC predefined names work if spelled like the enum.",
                GH_ParamAccess.item, "Column");
            pManager.AddTextParameter  ("Material",      "Mat",       "Material name to assign.",                             GH_ParamAccess.item, "Steel");
            pManager.AddTextParameter  ("FilePath",      "File",      "Output .ifc file path.",                               GH_ParamAccess.item);
            pManager.AddTextParameter  ("Schema",        "Schema",    "IFC schema: IFC2x3 or IFC4 (default).",                GH_ParamAccess.item, "IFC4");
            pManager.AddTextParameter  ("ProjectName",   "Project",   "IFC project name.",                                    GH_ParamAccess.item, "StructuralCircle Project");
            pManager.AddTextParameter  ("BuildingName",  "Building",  "IFC building name.",                                   GH_ParamAccess.item, "Building");
            pManager.AddBooleanParameter("Run",          "Run",       "Set to true to trigger export.",                       GH_ParamAccess.item, false);

            pManager[1].Optional = true;
            pManager[3].Optional = true;
            pManager[5].Optional = true;
            pManager[6].Optional = true;
            pManager[7].Optional = true;

            if (pManager[1] is Param_String pNames)
                pNames.SetPersistentData(new GH_String("element1"));
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Report",   "Report", "Export summary.",        GH_ParamAccess.item);
            pManager.AddTextParameter("FilePath", "File",   "Written IFC file path.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var breps     = new List<Brep>();
            var namesRaw  = new List<string>();
            string typStr = "Column";
            string matStr = "Steel";
            string file   = "";
            string schema = "IFC4";
            string proj   = "StructuralCircle Project";
            string bldg   = "Building";
            bool   run    = false;

            if (!DA.GetDataList(0, breps))  return;
            DA.GetDataList(1, namesRaw);
            if (!DA.GetData(2, ref typStr)) typStr = "Column";
            DA.GetData(3, ref matStr);
            if (!DA.GetData(4, ref file))   return;
            if (string.IsNullOrWhiteSpace(file))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "File path is empty.");
                return;
            }
            if (!DA.GetData(5, ref schema)) schema = "IFC4";
            if (!DA.GetData(6, ref proj))   proj   = "StructuralCircle Project";
            if (!DA.GetData(7, ref bldg))   bldg   = "Building";
            if (!DA.GetData(8, ref run))    return;

            if (!run) { DA.SetData(0, "Set Run to true to export."); return; }

            var names = new List<string>();
            for (int i = 0; i < breps.Count; i++)
            {
                if (i < namesRaw.Count && !string.IsNullOrWhiteSpace(namesRaw[i]))
                    names.Add(namesRaw[i].Trim());
                else
                    names.Add("element" + (i + 1));
            }

            if (!IfcBrepExporter.TryResolveExportElementType(typStr,
                    out IfcBrepExporter.StructuralType elemType,
                    out bool isStructural,
                    out IfcMechanicalFastenerTypeEnum fastenerType,
                    out bool isFastener))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    $"Unknown ElementType '{typStr}'. Structural: {string.Join(", ", StructuralOptions)}. "
                    + $"Fasteners (IfcMechanicalFastener): {string.Join(", ", FastenerOptions)}. "
                    + "Other predefined types from IfcMechanicalFastenerTypeEnum also work if spelled alike (non-letters are stripped).");
                return;
            }

            try
            {
                string report;
                if (isStructural)
                    report = IfcBrepExporter.Export(breps, names, elemType, matStr, file, schema, proj, bldg);
                else
                    report = IfcBrepExporter.ExportMechanicalFasteners(
                        breps, names, fastenerType, matStr, file, schema, proj, bldg);
                DA.SetData(0, report);
                DA.SetData(1, file);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
            }
        }
    }
}
