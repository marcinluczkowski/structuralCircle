using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    public class Construct_Material : GH_Component
    {
        public Construct_Material()
          : base("Construct Material", "MatCon",
              "Construct a Material from an ID and name",
              "StructuralCircleNTNU", "Construct")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Material identifier", GH_ParamAccess.item, 0);
            pManager.AddTextParameter("Name", "N", "Material name", GH_ParamAccess.item, "Timber");
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Material", "M", "Constructed Material", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int id = 0;
            string name = "";
            DA.GetData(0, ref id);
            DA.GetData(1, ref name);

            var material = new Material(id, name);
            DA.SetData(0, material);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000001");
    }
}
