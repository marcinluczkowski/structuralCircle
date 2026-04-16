using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.IFC
{
    public class Export_Elements_IFC : GH_Component
    {
        public Export_Elements_IFC()
            : base("Export Elements to IFC", "ExportIFC",
                   "Export StructuralCircle Beam / Plate elements to an IFC file (IFC2x3 or IFC4).",
                   "StructuralCircleNTNU", "IFC") { }

        public override Guid ComponentGuid => new Guid("a1b2c3d4-e5f6-7890-abcd-ef1234567801");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Elements",      "Elems",    "List of Beam / Plate elements.",           GH_ParamAccess.list);
            pManager.AddTextParameter   ("FilePath",      "File",     "Output IFC file path (.ifc).",             GH_ParamAccess.item);
            pManager.AddTextParameter   ("Schema",        "Schema",   "IFC schema: IFC2x3 or IFC4 (default).",    GH_ParamAccess.item, "IFC4");
            pManager.AddTextParameter   ("ProjectName",   "Project",  "IFC project name.",                        GH_ParamAccess.item, "StructuralCircle Project");
            pManager.AddTextParameter   ("BuildingName",  "Building", "IFC building name.",                       GH_ParamAccess.item, "Building");
            pManager.AddBooleanParameter("Run",           "Run",      "Set to true to export.",                   GH_ParamAccess.item, false);

            pManager[2].Optional = true;
            pManager[3].Optional = true;
            pManager[4].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Report",   "Report", "Export summary.",     GH_ParamAccess.item);
            pManager.AddTextParameter("FilePath", "File",   "Written IFC file path.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var rawList = new List<object>();
            string filePath    = "";
            string schema      = "IFC4";
            string projectName = "StructuralCircle Project";
            string building    = "Building";
            bool run           = false;

            if (!DA.GetDataList(0, rawList))    return;
            if (!DA.GetData(1, ref filePath))   return;
            if (string.IsNullOrWhiteSpace(filePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "File path is empty.");
                return;
            }
            if (!DA.GetData(2, ref schema))      schema = "IFC4";
            if (!DA.GetData(3, ref projectName)) projectName = "StructuralCircle Project";
            if (!DA.GetData(4, ref building))    building = "Building";
            if (!DA.GetData(5, ref run))         return;

            if (!run) { DA.SetData(0, "Set Run to true to export."); return; }

            var elements = new List<Element>();
            foreach (var r in rawList)
            {
                if (r is Element e) elements.Add(e);
            }

            if (elements.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid elements in list.");
                return;
            }

            try
            {
                string report = IfcStructuralExporter.Export(elements, filePath, schema, projectName, building);
                DA.SetData(0, report);
                DA.SetData(1, filePath);
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
            }
        }
    }
}
