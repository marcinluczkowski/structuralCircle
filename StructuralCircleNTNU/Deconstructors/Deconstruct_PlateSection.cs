using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_PlateSection : GH_Component
    {
        public Deconstruct_PlateSection()
            : base("Deconstruct Plate Section", "DeconPSec",
                   "Explode a PlateSection into its properties.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001003-0000-0000-0000-000000000003");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("PlateSection", "PSec", "PlateSection to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id",        "Id",  "Section id.",          GH_ParamAccess.item);
            pManager.AddTextParameter   ("Name",      "Name","Section name.",         GH_ParamAccess.item);
            pManager.AddNumberParameter ("Thickness", "T",   "Thickness (m).",        GH_ParamAccess.item);
            pManager.AddNumberParameter ("Width",     "W",   "Panel width (m).",      GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var s = raw as PlateSection;
            if (s == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a PlateSection."); return; }

            DA.SetData(0, s.Id);
            DA.SetData(1, s.Name);
            DA.SetData(2, s.Thickness);
            DA.SetData(3, s.Width);
        }
    }
}
