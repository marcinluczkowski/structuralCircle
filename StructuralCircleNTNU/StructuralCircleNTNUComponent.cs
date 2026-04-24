using System;
using Grasshopper.Kernel;

namespace StructuralCircleNTNU
{
    public class StructuralCircleNTNUComponent : GH_Component
    {
        public StructuralCircleNTNUComponent()
          : base("StructuralCircle Info", "SCInfo",
            "StructuralCircle NTNU - Sustainable design from used structural elements. " +
            "Matching algorithms for reclaimed building components.",
            "StructuralCircleNTNU", "Info")
        { }
        //
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Info", "I", "Plugin information", GH_ParamAccess.item);
            pManager.AddTextParameter("Version", "V", "Plugin version", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            DA.SetData(0, "StructuralCircle NTNU - Matching algorithms for reclaimed building components.");
            DA.SetData(1, GetType().Assembly.GetName().Version.ToString());
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("645c4206-fb4e-4810-9da6-15c7a5696b81");
    }
}
