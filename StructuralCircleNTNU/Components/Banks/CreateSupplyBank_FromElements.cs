using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Banks
{
    public class CreateSupplyBank_FromElements : GH_Component
    {
        public CreateSupplyBank_FromElements()
          : base("Supply Bank (Elements)", "SBank",
              "Create a Supply Bank from a list of elements",
              "StructuralCircleNTNU", "Banks")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Elements", "E", "List of supply elements (Beams and/or Plates)", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "SB", "Constructed Supply Bank", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var elements = new List<Element>();
            if (!DA.GetDataList(0, elements)) return;

            var bank = new SupplyBank(elements);
            DA.SetData(0, bank);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000010");
    }
}
