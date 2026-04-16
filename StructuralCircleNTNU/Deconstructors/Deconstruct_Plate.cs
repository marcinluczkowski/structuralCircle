using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_Plate : GH_Component
    {
        public Deconstruct_Plate()
            : base("Deconstruct Plate", "DeconPlate",
                   "Explode a Plate into its properties.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001005-0000-0000-0000-000000000005");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Plate", "Plate", "Plate to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id",       "Id",    "Element id.",          GH_ParamAccess.item);
            pManager.AddTextParameter   ("Name",     "Name",  "Element name.",         GH_ParamAccess.item);
            pManager.AddTextParameter   ("Location", "Loc",   "Location string.",      GH_ParamAccess.item);
            pManager.AddGenericParameter("Material", "Mat",   "Material object.",      GH_ParamAccess.item);
            pManager.AddGenericParameter("Section",  "Sec",   "PlateSection object.",  GH_ParamAccess.item);
            pManager.AddNumberParameter ("Length",   "L",     "Length (m).",           GH_ParamAccess.item);
            pManager.AddBrepParameter   ("Brep",     "Brep",  "Geometry Brep.",        GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var plate = raw as Plate;
            if (plate == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a Plate."); return; }

            DA.SetData(0, plate.Id);
            DA.SetData(1, plate.Name);
            DA.SetData(2, plate.Location);
            DA.SetData(3, plate.Material);
            DA.SetData(4, plate.Section);
            DA.SetData(5, plate.Length);
            if (plate.GeometryBrep != null) DA.SetData(6, plate.GeometryBrep);
        }
    }
}
