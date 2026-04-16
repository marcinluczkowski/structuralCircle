using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    public class Construct_Plate : GH_Component
    {
        public Construct_Plate()
          : base("Construct Plate", "PlCon",
              "Construct a Plate element from basic inputs",
              "StructuralCircleNTNU", "Construct")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Element identifier", GH_ParamAccess.item, 0);
            pManager.AddTextParameter("Name", "N", "Element name", GH_ParamAccess.item, "Plate_01");
            pManager.AddTextParameter("Location", "Loc", "Location description", GH_ParamAccess.item, "");
            pManager.AddGenericParameter("Material", "M", "Material", GH_ParamAccess.item);
            pManager.AddGenericParameter("Section", "S", "PlateSection", GH_ParamAccess.item);
            pManager.AddSurfaceParameter("Surface", "Srf", "Plate axis surface", GH_ParamAccess.item);
            pManager.AddBrepParameter("Brep", "B", "Optional 3D Brep geometry", GH_ParamAccess.item);
            pManager.AddMeshParameter("Mesh", "Me", "Optional mesh geometry", GH_ParamAccess.item);
            pManager[2].Optional = true;
            pManager[5].Optional = true;
            pManager[6].Optional = true;
            pManager[7].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Plate", "P", "Constructed Plate element", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int id = 0;
            string name = "", location = "";
            Material material = null;
            PlateSection section = null;
            Surface surface = null;
            Brep brep = null;
            Mesh mesh = null;

            DA.GetData(0, ref id);
            DA.GetData(1, ref name);
            DA.GetData(2, ref location);
            DA.GetData(3, ref material);
            DA.GetData(4, ref section);
            DA.GetData(5, ref surface);
            DA.GetData(6, ref brep);
            DA.GetData(7, ref mesh);

            if (material == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Material is required");
                return;
            }
            if (section == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "PlateSection is required");
                return;
            }

            var plate = new Plate(id, name, location, material, section, surface);
            if (brep != null) plate.GeometryBrep = brep;
            if (mesh != null) plate.GeometryMesh = mesh;

            DA.SetData(0, plate);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000005");
    }
}
