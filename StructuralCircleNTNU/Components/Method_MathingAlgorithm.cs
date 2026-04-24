using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components
{
    public class Method_MatchingAlgorithm : GH_Component
    {
        public Method_MatchingAlgorithm()
          : base("Matching Algorithm", "Match",
              "Match supply to demand. Mode 0=Greedy, 1=BruteForce (subset + combination cap), " +
              "2=MILP (bipartite LP via Hungarian = LP optimum), 3=GreedyPacking (cut-from-stock, " +
              "multiple demand per supply), 4=MilpPacking (BFD + local search). " +
              "Connect Run to a Button or True to execute; when False the component skips work so Grasshopper stays responsive.",
              "StructuralCircleNTNU", "Matching")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "SB", "Supply Bank of available elements", GH_ParamAccess.item);
            pManager.AddGenericParameter("DemandBank", "DB", "Demand Bank of required elements", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Mode", "M",
                "0 = Greedy (fast one-pass). " +
                "1 = BruteForce (exact on a limited demand subset). " +
                "2 = MILP (Mixed-Integer LP bipartite matching; Hungarian = LP optimum). " +
                "3 = GreedyPacking (FFD cutting-stock; multiple demand per supply). " +
                "4 = MilpPacking (BFD + local-search packing).",
                GH_ParamAccess.item, 1);
            pManager.AddBooleanParameter("Run", "Run",
                "If False, matching is skipped (no CPU load). If this input is not wired, behaves as True for backward compatibility. Connect a Button for a run trigger.",
                GH_ParamAccess.item, false);
            pManager.AddNumberParameter("DemandSubset", "D%",
                "BruteForce only: fraction (0–1] of demand rows to optimize (from the start of the bank). Default 0.1. Ignored if MaxDemand > 0.",
                GH_ParamAccess.item, 0.1);
            pManager.AddIntegerParameter("MaxDemand", "kD",
                "BruteForce only: if > 0, maximum number of leading demand elements in the search; overrides DemandSubset. 0 = use DemandSubset.",
                GH_ParamAccess.item, 0);
            pManager.AddIntegerParameter("PackingMode", "PM",
                "Packing modes (3 and 4) only: 0 = Auto (1D for beams, 2D for plates, 3D otherwise), " +
                "1 = 1D cutting stock along length, 2 = 2D shelf packing on plate area, " +
                "3 = 3D BBox volume packing, 4 = Brep (falls back to 3D BBox).",
                GH_ParamAccess.item, 0);

            pManager[3].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Result", "R", "Matching result", GH_ParamAccess.item);
            pManager.AddTextParameter("Report", "Rpt", "Text report of matching results", GH_ParamAccess.item);
            pManager.AddGenericParameter("Matched Demand", "MD", "Matched demand elements", GH_ParamAccess.list);
            pManager.AddGenericParameter("Matched Supply", "MS", "Matched supply elements", GH_ParamAccess.list);
            pManager.AddGenericParameter("Unmatched Demand", "UD", "Unmatched demand elements", GH_ParamAccess.list);
            pManager.AddGenericParameter("Unmatched Supply", "US", "Unmatched supply elements", GH_ParamAccess.list);
            pManager.AddNumberParameter("Scores", "Sc", "Score for each matched pair", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object rawSupply = null;
            object rawDemand = null;
            if (!DA.GetData(0, ref rawSupply)) return;
            if (!DA.GetData(1, ref rawDemand)) return;

            var supply = GrasshopperUnpack.AsSupplyBank(rawSupply);
            var demand = GrasshopperUnpack.AsDemandBank(rawDemand);

            int mode = 1;
            DA.GetData(2, ref mode);

            bool run = true;
            if (!DA.GetData(3, ref run))
                run = true;

            double demandSubset = 0.1;
            if (!DA.GetData(4, ref demandSubset))
                demandSubset = 0.1;

            int maxDemand = 0;
            DA.GetData(5, ref maxDemand);

            int packingModeInt = 0;
            DA.GetData(6, ref packingModeInt);
            PackingMode packingMode = Enum.IsDefined(typeof(PackingMode), packingModeInt)
                ? (PackingMode)packingModeInt
                : PackingMode.Auto;

            if (supply == null || demand == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Supply and Demand banks are required");
                return;
            }

            if (!run)
            {
                DA.SetData(0, null);
                DA.SetData(1, "Run is False — set True or click a connected Button to compute matching.");
                DA.SetDataList(2, new List<Element>());
                DA.SetDataList(3, new List<Element>());
                DA.SetDataList(4, new List<Element>());
                DA.SetDataList(5, new List<Element>());
                DA.SetDataList(6, new List<double>());
                return;
            }

            if (demand.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Demand bank is empty");
                return;
            }

            if (supply.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Supply bank is empty");
                return;
            }

            if (mode == (int)MatchingAlgorithmMode.BruteForce)
            {
                int k = MatchingEngine.ResolveBruteDemandCount(demand.Count, demandSubset, maxDemand);
                int combinationLimit = 1;
                for (int i = 0; i < k; i++)
                {
                    long rowChoices = supply.Count + 1;
                    combinationLimit = (int)Math.Min((long)combinationLimit * rowChoices, int.MaxValue);
                }

                if (combinationLimit > 1_000_000)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                        $"BruteForce considers {k} demand row(s). Estimated raw combinations up to (S+1)^k; execution is blocked above {MatchingEngine.DefaultMaxBruteCombinations:N0}.");
                }
            }

            MatchResult result;
            switch (mode)
            {
                case (int)MatchingAlgorithmMode.Greedy:
                    result = MatchingEngine.GreedyMatch(demand, supply);
                    break;
                case (int)MatchingAlgorithmMode.BruteForce:
                    result = MatchingEngine.BruteForceMatch(demand, supply, demandSubset, maxDemand);
                    if (result.Note != null && result.Note.IndexOf("skipped", StringComparison.OrdinalIgnoreCase) >= 0)
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, result.Note);
                    else if (!string.IsNullOrEmpty(result.Note))
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, result.Note);
                    break;
                case (int)MatchingAlgorithmMode.Milp:
                    result = MatchingEngine.MilpMatch(demand, supply);
                    if (!string.IsNullOrEmpty(result.Note))
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, result.Note);
                    break;
                case (int)MatchingAlgorithmMode.GreedyPacking:
                    result = MatchingEngine.GreedyPackingMatch(demand, supply, packingMode);
                    if (!string.IsNullOrEmpty(result.Note))
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, result.Note);
                    break;
                case (int)MatchingAlgorithmMode.MilpPacking:
                    result = MatchingEngine.MilpPackingMatch(demand, supply, packingMode);
                    if (!string.IsNullOrEmpty(result.Note))
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, result.Note);
                    break;
                default:
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Mode {mode} is not defined (use 0–4). Using Greedy.");
                    result = MatchingEngine.GreedyMatch(demand, supply);
                    break;
            }

            DA.SetData(0, result);
            DA.SetData(1, result.ToString());

            var matchedDemand = new List<Element>();
            var matchedSupply = new List<Element>();
            var scores = new List<double>();
            if (result.Pairs != null)
            {
                foreach (var pair in result.Pairs)
                {
                    matchedDemand.Add(pair.Demand);
                    matchedSupply.Add(pair.Supply);
                    scores.Add(pair.Score);
                }
            }

            DA.SetDataList(2, matchedDemand);
            DA.SetDataList(3, matchedSupply);
            DA.SetDataList(4, result.UnmatchedDemand ?? new List<Element>());
            DA.SetDataList(5, result.UnmatchedSupply ?? new List<Element>());
            DA.SetDataList(6, scores);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("14BD2B84-BF33-48BE-9F97-EC37DC1F898F");
    }
}
