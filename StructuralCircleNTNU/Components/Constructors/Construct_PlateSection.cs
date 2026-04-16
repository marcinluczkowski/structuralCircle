using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    public class Construct_PlateSection : GH_Component
    {
        public Construct_PlateSection()
          : base("Construct Plate Section", "PlSec",
              "Construct a plate section from thickness",
              "StructuralCircleNTNU", "Construct")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Section identifier", GH_ParamAccess.item, 0);
            pManager.AddTextParameter("Name", "N", "Section name", GH_ParamAccess.item, "Plate_20mm");
            pManager.AddNumberParameter("Thickness", "T", "Plate thickness [m]", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Section", "S", "Constructed PlateSection", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int id = 0;
            string name = "";
            double thickness = 0;
            DA.GetData(0, ref id);
            DA.GetData(1, ref name);
            DA.GetData(2, ref thickness);

            var section = new PlateSection(id, name, thickness);
            DA.SetData(0, section);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000003");
    }
}
