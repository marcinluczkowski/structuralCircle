using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_MatchPair : GH_Component
    {
        public Deconstruct_MatchPair()
            : base("Deconstruct Match Pair", "DeconPair",
                   "Explode a MatchPair into supply/demand elements and score.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001008-0000-0000-0000-000000000008");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("MatchPair", "Pair", "MatchPair to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyElement", "Supply", "Matched supply element.", GH_ParamAccess.item);
            pManager.AddGenericParameter("DemandElement", "Demand", "Matched demand element.", GH_ParamAccess.item);
            pManager.AddNumberParameter ("Score",         "Score",  "Match quality score.",    GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var pair = raw as MatchPair;
            if (pair == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a MatchPair."); return; }

            DA.SetData(0, pair.Supply);
            DA.SetData(1, pair.Demand);
            DA.SetData(2, pair.Score);
        }
    }
}
