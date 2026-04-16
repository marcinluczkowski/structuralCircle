using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_BeamSection : GH_Component
    {
        public Deconstruct_BeamSection()
            : base("Deconstruct Beam Section", "DeconBSec",
                   "Explode a BeamSection into its properties.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001002-0000-0000-0000-000000000002");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("BeamSection", "BSec", "BeamSection to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id",     "Id",     "Section id.",         GH_ParamAccess.item);
            pManager.AddTextParameter   ("Name",   "Name",   "Section name.",        GH_ParamAccess.item);
            pManager.AddNumberParameter ("Width",  "W",      "Width (m).",           GH_ParamAccess.item);
            pManager.AddNumberParameter ("Height", "H",      "Height (m).",          GH_ParamAccess.item);
            pManager.AddNumberParameter ("Area",   "A",      "Area (m²).",           GH_ParamAccess.item);
            pManager.AddNumberParameter ("Iy",     "Iy",     "Second moment Iy.",    GH_ParamAccess.item);
            pManager.AddNumberParameter ("Iz",     "Iz",     "Second moment Iz.",    GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var s = raw as BeamSection;
            if (s == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a BeamSection."); return; }

            DA.SetData(0, s.Id);
            DA.SetData(1, s.Name);
            DA.SetData(2, s.Width);
            DA.SetData(3, s.Height);
            DA.SetData(4, s.Area);
            DA.SetData(5, s.Iy);
            DA.SetData(6, s.Iz);
        }
    }
}
