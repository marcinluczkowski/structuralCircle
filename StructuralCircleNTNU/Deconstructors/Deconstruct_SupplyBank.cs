using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_SupplyBank : GH_Component
    {
        public Deconstruct_SupplyBank()
            : base("Deconstruct Supply Bank", "DeconSupply",
                   "Explode a SupplyBank into its elements.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001006-0000-0000-0000-000000000006");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "Supply", "Supply bank to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Elements", "Elems", "All elements.",  GH_ParamAccess.list);
            pManager.AddGenericParameter("Beams",    "Beams", "Beam elements.", GH_ParamAccess.list);
            pManager.AddGenericParameter("Plates",   "Plates","Plate elements.",GH_ParamAccess.list);
            pManager.AddIntegerParameter("Count",    "N",     "Total count.",   GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var bank = GrasshopperUnpack.AsSupplyBank(raw);
            if (bank == null)
            {
                if (GrasshopperUnpack.AsDemandBank(raw) != null)
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                        "Input is a DemandBank. Use the Deconstruct Demand Bank component.");
                else
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                        "Input is not a SupplyBank (Grasshopper may wrap it; if the wire is correct, try re-internalizing the upstream component).");
                return;
            }

            DA.SetDataList(0, bank.Elements);
            DA.SetDataList(1, bank.Beams);
            DA.SetDataList(2, bank.Plates);
            DA.SetData    (3, bank.Elements.Count);
        }
    }
}
