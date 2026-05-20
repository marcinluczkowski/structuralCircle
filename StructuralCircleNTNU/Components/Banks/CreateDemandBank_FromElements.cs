using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Banks
{
    public class CreateDemandBank_FromElements : GH_Component
    {
        public CreateDemandBank_FromElements()
          : base("Demand Bank (Elements)", "DBank",
              "Create a Demand Bank from a list of elements",
              "StructuralCircleNTNU", "Banks")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Elements", "E", "List of demand elements (Beams and/or Plates)", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("DemandBank", "DB", "Constructed Demand Bank", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var rawList = new List<object>();
            if (!DA.GetDataList(0, rawList)) return;

            var elements = new List<Element>();
            foreach (var r in rawList)
            {
                var e = GrasshopperUnpack.AsElement(r);
                if (e != null) elements.Add(e);
            }

            if (elements.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid elements in list.");
                return;
            }

            var bank = new DemandBank(elements);
            DA.SetData(0, bank);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000011");
    }
}
