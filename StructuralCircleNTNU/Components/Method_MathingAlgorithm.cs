using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components
{
    public class Method_MatchingAlgorithm : GH_Component
    {
        public Method_MatchingAlgorithm()
          : base("Matching Algorithm", "Match",
              "Match supply elements to demand elements using brute force optimization. " +
              "Based on the structuralCircle algorithm from NTNU.",
              "StructuralCircleNTNU", "Matching")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "SB", "Supply Bank of available elements", GH_ParamAccess.item);
            pManager.AddGenericParameter("DemandBank", "DB", "Demand Bank of required elements", GH_ParamAccess.item);
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
            SupplyBank supply = null;
            DemandBank demand = null;

            if (!DA.GetData(0, ref supply)) return;
            if (!DA.GetData(1, ref demand)) return;

            if (supply == null || demand == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Supply and Demand banks are required");
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

            int combinationLimit = 1;
            foreach (var _ in demand.Elements)
                combinationLimit *= (supply.Count + 1);

            if (combinationLimit > 1_000_000)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"Large problem ({demand.Count} demand x {supply.Count} supply). Brute force may be slow.");
            }

            var result = MatchingEngine.BruteForceMatch(demand, supply);

            DA.SetData(0, result);
            DA.SetData(1, result.ToString());

            var matchedDemand = new List<Element>();
            var matchedSupply = new List<Element>();
            var scores = new List<double>();
            foreach (var pair in result.Pairs)
            {
                matchedDemand.Add(pair.Demand);
                matchedSupply.Add(pair.Supply);
                scores.Add(pair.Score);
            }

            DA.SetDataList(2, matchedDemand);
            DA.SetDataList(3, matchedSupply);
            DA.SetDataList(4, result.UnmatchedDemand);
            DA.SetDataList(5, result.UnmatchedSupply);
            DA.SetDataList(6, scores);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("14BD2B84-BF33-48BE-9F97-EC37DC1F898F");
    }
}
