using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.IFC
{
    /// <summary>
    /// Takes arbitrary Rhino Breps and exports them as structural IFC elements.
    /// Geometry is tessellated to IfcFacetedBrep.
    /// Supported IFC types: Beam, Column, Wall, Slab, Member.
    /// </summary>
    public class Export_Brep_IFC : GH_Component
    {
        private static readonly string[] TypeOptions =
            { "Beam", "Column", "Wall", "Slab", "Member" };

        public Export_Brep_IFC()
            : base("Brep to IFC", "BrepIFC",
                   "Export Rhino Breps as structural IFC elements (Beam, Column, Wall, Slab, Member).",
                   "StructuralCircleNTNU", "IFC") { }

        public override Guid ComponentGuid => new Guid("b2c3d4e5-f6a7-8901-bcde-f12345678902");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter  ("Breps",         "Breps",     "Geometry to export.",                                  GH_ParamAccess.list);
            pManager.AddTextParameter  ("Names",         "Names",     "Element names (one per Brep). Default: element1, element2, …", GH_ParamAccess.list);
            pManager.AddTextParameter  ("ElementType",   "Type",      "IFC type: Beam | Column | Wall | Slab | Member.",      GH_ParamAccess.item, "Column");
            pManager.AddTextParameter  ("Material",      "Mat",       "Material name to assign.",                             GH_ParamAccess.item, "Timber");
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
            string matStr = "Timber";
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

            if (!Enum.TryParse(typStr, true, out IfcBrepExporter.StructuralType elemType))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    $"Unknown ElementType '{typStr}'. Choose from: {string.Join(", ", TypeOptions)}");
                return;
            }

            try
            {
                string report = IfcBrepExporter.Export(breps, names, elemType, matStr, file, schema, proj, bldg);
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
