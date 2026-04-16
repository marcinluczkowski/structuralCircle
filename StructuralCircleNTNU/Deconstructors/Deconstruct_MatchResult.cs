using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_MatchResult : GH_Component
    {
        public Deconstruct_MatchResult()
            : base("Deconstruct Match Result", "DeconResult",
                   "Explode a MatchResult into pairs, unmatched elements, total score, and method.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("DE001009-0000-0000-0000-000000000009");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("MatchResult", "Result", "MatchResult to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Pairs",            "Pairs",     "All MatchPair objects.",          GH_ParamAccess.list);
            pManager.AddGenericParameter("UnmatchedSupply",  "UnSup",     "Unmatched supply elements.",      GH_ParamAccess.list);
            pManager.AddGenericParameter("UnmatchedDemand",  "UnDem",     "Unmatched demand elements.",      GH_ParamAccess.list);
            pManager.AddNumberParameter ("TotalScore",       "Score",     "Total weighted match score.",     GH_ParamAccess.item);
            pManager.AddTextParameter   ("Method",           "Method",    "Matching method name.",           GH_ParamAccess.item);
            pManager.AddTextParameter   ("Report",           "Report",    "Human-readable summary.",         GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var result = raw as MatchResult;
            if (result == null) { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a MatchResult."); return; }

            DA.SetDataList(0, result.Pairs);
            DA.SetDataList(1, result.UnmatchedSupply);
            DA.SetDataList(2, result.UnmatchedDemand);
            DA.SetData    (3, result.TotalScore);
            DA.SetData    (4, result.Method);
            DA.SetData    (5, result.ToString());
        }
    }
}
