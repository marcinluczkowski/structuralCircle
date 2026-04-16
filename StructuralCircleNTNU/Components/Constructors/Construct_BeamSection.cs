using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    public class Construct_BeamSection : GH_Component
    {
        public Construct_BeamSection()
          : base("Construct Beam Section", "BmSec",
              "Construct a rectangular beam section from width and height. Iy and Iz are computed automatically if not provided.",
              "StructuralCircleNTNU", "Construct")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Section identifier", GH_ParamAccess.item, 0);
            pManager.AddTextParameter("Name", "N", "Section name", GH_ParamAccess.item, "200x300");
            pManager.AddNumberParameter("Width", "W", "Section width [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("Height", "H", "Section height [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("Iy", "Iy", "Moment of inertia about Y axis [m^4]. Auto-computed if not set.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Iz", "Iz", "Moment of inertia about Z axis [m^4]. Auto-computed if not set.", GH_ParamAccess.item);
            pManager[4].Optional = true;
            pManager[5].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Section", "S", "Constructed BeamSection", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int id = 0;
            string name = "";
            double width = 0, height = 0;
            DA.GetData(0, ref id);
            DA.GetData(1, ref name);
            DA.GetData(2, ref width);
            DA.GetData(3, ref height);

            double iy = 0, iz = 0;
            bool hasIy = DA.GetData(4, ref iy);
            bool hasIz = DA.GetData(5, ref iz);

            BeamSection section;
            if (hasIy && hasIz)
                section = new BeamSection(id, name, width, height, iy, iz);
            else
                section = new BeamSection(id, name, width, height);

            DA.SetData(0, section);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000002");
    }
}
