using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_Beam : GH_Component
    {
        public Deconstruct_Beam()
            : base("Deconstruct Beam", "DeconBeam",
                   "Explode a Beam into its properties.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001004-0000-0000-0000-000000000004");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Beam", "Beam", "Beam to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id",       "Id",    "Element id.",        GH_ParamAccess.item);
            pManager.AddTextParameter   ("Name",     "Name",  "Element name.",       GH_ParamAccess.item);
            pManager.AddTextParameter   ("Location", "Loc",   "Location string.",    GH_ParamAccess.item);
            pManager.AddGenericParameter("Material", "Mat",   "Material object.",    GH_ParamAccess.item);
            pManager.AddGenericParameter("Section",  "Sec",   "BeamSection object.", GH_ParamAccess.item);
            pManager.AddLineParameter   ("AxisLine", "Axis",  "Axis line.",          GH_ParamAccess.item);
            pManager.AddNumberParameter ("Length",   "L",     "Length (m).",         GH_ParamAccess.item);
            pManager.AddBrepParameter   ("Brep",     "Brep",  "Geometry Brep.",      GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var beam = raw as Beam;
            if (beam == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a Beam."); return; }

            DA.SetData(0, beam.Id);
            DA.SetData(1, beam.Name);
            DA.SetData(2, beam.Location);
            DA.SetData(3, beam.Material);
            DA.SetData(4, beam.Section);
            DA.SetData(5, beam.AxisLine);
            DA.SetData(6, beam.Length);
            if (beam.GeometryBrep != null) DA.SetData(7, beam.GeometryBrep);
        }
    }
}
