using System;
using System.Collections.Generic;
using System.Text;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components
{
    /// <summary>
    /// Grasshopper component that runs one of three matching algorithms to pair supply elements
    /// with demand elements while minimising total waste volume (supply_vol − demand_vol).
    ///
    /// Method input:
    ///   0 = Brute Force  — exhaustive, exact, O((nS+1)^nD). Only for small banks (≤ ~8 elements each).
    ///   1 = Greedy       — fast heuristic, sorts demand by volume desc, picks minimum-waste supply.
    ///                      O(nD × nS). Not globally optimal but scales to any size.
    ///   2 = Hungarian    — optimal bipartite assignment, O(n³). Recommended for medium/large banks.
    ///                      Equivalent to the bipartite method in structuralCircle Python repo.
    ///
    /// Constraints (supply must satisfy demand):
    ///   Beams  → Length ≥, section Area ≥, Iy ≥, Iz ≥
    ///   Plates → Thickness ≥ (and Length ≥ when both are defined)
    /// </summary>
    public class Method_MatchingAlgorithm : GH_Component
    {
        public Method_MatchingAlgorithm()
          : base("Matching Algorithm", "Match",
              "Match supply elements to demand elements minimising total waste volume.\n" +
              "Method: 0 = Brute Force (exact, small banks only) | 1 = Greedy (fast) | 2 = Hungarian (optimal, default)",
              "StructuralCircleNTNU", "Matching")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "SB",  "Supply Bank of available elements.",                                                  GH_ParamAccess.item);
            pManager.AddGenericParameter("DemandBank", "DB",  "Demand Bank of required elements.",                                                   GH_ParamAccess.item);
            pManager.AddIntegerParameter("Method",     "M",   "Algorithm: 0=BruteForce  1=Greedy  2=Hungarian (default).",                           GH_ParamAccess.item, 2);
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Result",          "R",   "Matching result object.",                                            GH_ParamAccess.item);
            pManager.AddTextParameter   ("Report",          "Rpt", "Detailed text report.",                                              GH_ParamAccess.item);
            pManager.AddGenericParameter("Matched Demand",  "MD",  "Matched demand elements (same order as Matched Supply).",            GH_ParamAccess.list);
            pManager.AddGenericParameter("Matched Supply",  "MS",  "Matched supply elements (same order as Matched Demand).",            GH_ParamAccess.list);
            pManager.AddGenericParameter("Unmatched Demand","UD",  "Demand elements that could not be satisfied by any supply.",         GH_ParamAccess.list);
            pManager.AddGenericParameter("Unmatched Supply","US",  "Supply elements not used in the best assignment.",                   GH_ParamAccess.list);
            pManager.AddNumberParameter ("WasteVolume",     "Wv",  "Waste volume per matched pair (supply_vol − demand_vol, m³).",       GH_ParamAccess.list);
            pManager.AddNumberParameter ("TotalWaste",      "TW",  "Total waste volume across all matched pairs (m³).",                  GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            SupplyBank supply = null;
            DemandBank demand = null;
            int method = 2;

            if (!DA.GetData(0, ref supply)) return;
            if (!DA.GetData(1, ref demand)) return;
            DA.GetData(2, ref method);

            if (supply == null || demand == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Supply and Demand banks are required.");
                return;
            }
            if (demand.Count == 0) { AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Demand bank is empty."); return; }
            if (supply.Count == 0) { AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Supply bank is empty."); return; }

            MatchResult result;
            switch (method)
            {
                case 0:
                    // Warn about exponential complexity before running.
                    long combEst = 1;
                    foreach (var _ in demand.Elements)
                    {
                        combEst *= (supply.Count + 1);
                        if (combEst > 1_000_000) break;
                    }
                    if (combEst > 1_000_000)
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                            $"Brute Force: {demand.Count} demand × {supply.Count} supply → search space may be very large. Consider Method 2 (Hungarian) instead.");
                    result = MatchingEngine.BruteForceMatch(demand, supply);
                    break;

                case 1:
                    result = MatchingEngine.GreedyMatch(demand, supply);
                    break;

                default:
                    if (method != 2)
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Unknown method {method}, defaulting to Hungarian (2).");
                    result = MatchingEngine.HungarianMatch(demand, supply);
                    break;
            }

            var matchedDemand = new List<Element>();
            var matchedSupply = new List<Element>();
            var wastePerPair  = new List<double>();
            foreach (var pair in result.Pairs)
            {
                matchedDemand.Add(pair.Demand);
                matchedSupply.Add(pair.Supply);
                wastePerPair.Add(pair.Score);
            }

            DA.SetData    (0, result);
            DA.SetData    (1, BuildReport(result, method));
            DA.SetDataList(2, matchedDemand);
            DA.SetDataList(3, matchedSupply);
            DA.SetDataList(4, result.UnmatchedDemand);
            DA.SetDataList(5, result.UnmatchedSupply);
            DA.SetDataList(6, wastePerPair);
            DA.SetData    (7, result.TotalScore);
        }

        static string BuildReport(MatchResult result, int methodIdx)
        {
            string[] methodLabels =
            {
                "0 — Brute Force (exhaustive, exact)",
                "1 — Greedy (heuristic, minimum-waste-first)",
                "2 — Hungarian / Bipartite (optimal, O(n³))"
            };
            string label = (methodIdx >= 0 && methodIdx < methodLabels.Length)
                ? methodLabels[methodIdx]
                : $"{methodIdx} — {result.Method}";

            var sb = new StringBuilder();
            sb.AppendLine($"=== Structural Circle Matching ===");
            sb.AppendLine($"  Method           : {label}");
            sb.AppendLine($"  Matched pairs    : {result.Pairs.Count}");
            sb.AppendLine($"  Unmatched demand : {result.UnmatchedDemand.Count}");
            sb.AppendLine($"  Unmatched supply : {result.UnmatchedSupply.Count}");
            sb.AppendLine($"  Total waste vol  : {result.TotalScore:F6} m³");

            if (result.Pairs.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("  Pairs (demand → supply | waste m³):");
                foreach (var p in result.Pairs)
                    sb.AppendLine($"    {p.Demand.Name,-26} → {p.Supply.Name,-26}  waste = {p.Score:F6} m³");
            }

            if (result.UnmatchedDemand.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("  Unmatched demand (no feasible supply found):");
                foreach (var e in result.UnmatchedDemand)
                    sb.AppendLine($"    {e}");
            }

            if (result.UnmatchedSupply.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("  Unused supply:");
                foreach (var e in result.UnmatchedSupply)
                    sb.AppendLine($"    {e}");
            }

            return sb.ToString();
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("14BD2B84-BF33-48BE-9F97-EC37DC1F898F");
    }
}
