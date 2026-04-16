using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    public class Construct_Beam : GH_Component
    {
        public Construct_Beam()
          : base("Construct Beam", "BmCon",
              "Construct a Beam element from basic inputs",
              "StructuralCircleNTNU", "Construct")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Element identifier", GH_ParamAccess.item, 0);
            pManager.AddTextParameter("Name", "N", "Element name", GH_ParamAccess.item, "Beam_01");
            pManager.AddTextParameter("Location", "Loc", "Location description", GH_ParamAccess.item, "");
            pManager.AddGenericParameter("Material", "M", "Material", GH_ParamAccess.item);
            pManager.AddGenericParameter("Section", "S", "BeamSection", GH_ParamAccess.item);
            pManager.AddLineParameter("Axis", "Ax", "Beam axis line", GH_ParamAccess.item);
            pManager.AddBrepParameter("Brep", "B", "Optional 3D Brep geometry", GH_ParamAccess.item);
            pManager.AddMeshParameter("Mesh", "Me", "Optional mesh geometry", GH_ParamAccess.item);
            pManager[2].Optional = true;
            pManager[6].Optional = true;
            pManager[7].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Beam", "B", "Constructed Beam element", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int id = 0;
            string name = "", location = "";
            Material material = null;
            BeamSection section = null;
            Line axis = Line.Unset;
            Brep brep = null;
            Mesh mesh = null;

            DA.GetData(0, ref id);
            DA.GetData(1, ref name);
            DA.GetData(2, ref location);
            DA.GetData(3, ref material);
            DA.GetData(4, ref section);
            DA.GetData(5, ref axis);
            DA.GetData(6, ref brep);
            DA.GetData(7, ref mesh);

            if (material == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Material is required");
                return;
            }
            if (section == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "BeamSection is required");
                return;
            }

            var beam = new Beam(id, name, location, material, section, axis);
            if (brep != null) beam.GeometryBrep = brep;
            if (mesh != null) beam.GeometryMesh = mesh;

            DA.SetData(0, beam);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000004");
    }
}
