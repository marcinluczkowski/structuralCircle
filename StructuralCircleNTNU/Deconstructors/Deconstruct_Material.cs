using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_Material : GH_Component
    {
        public Deconstruct_Material()
            : base("Deconstruct Material", "DeconMat",
                   "Explode a Material into its properties.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001001-0000-0000-0000-000000000001");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Material", "Mat", "Material to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id",   "Id",   "Material id.",   GH_ParamAccess.item);
            pManager.AddTextParameter   ("Name", "Name", "Material name.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var mat = GrasshopperUnpack.AsMaterial(raw);
            if (mat == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a Material."); return; }

            DA.SetData(0, mat.Id);
            DA.SetData(1, mat.Name);
        }
    }
}
