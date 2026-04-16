using System;
using System.Collections.Generic;
using System.Linq;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Constraint definition: supply property must satisfy operator against demand property.
    /// E.g. supply.Area >= demand.Area
    /// </summary>
    public class MatchConstraint
    {
        public string PropertyName { get; set; }
        public string Operator { get; set; }

        public MatchConstraint(string propertyName, string op)
        {
            PropertyName = propertyName;
            Operator = op;
        }
    }

    public static class MatchingEngine
    {
        /// <summary>
        /// Brute force matching: enumerate all valid one-to-one assignments and pick the best.
        /// Translated from structuralCircle Python repo (match_brute + extract_brute_possibilities).
        /// </summary>
        public static MatchResult BruteForceMatch(DemandBank demand, SupplyBank supply)
        {
            var demandElements = demand.Elements;
            var supplyElements = supply.Elements;

            int nDemand = demandElements.Count;
            int nSupply = supplyElements.Count;

            bool[,] incidence = EvaluateIncidence(demandElements, supplyElements);
            double[,] weights = EvaluateWeights(demandElements, supplyElements);

            var possibleAssignments = ExtractBrutePossibilities(incidence, nDemand, nSupply);

            MatchResult bestResult = null;
            double bestScore = double.MaxValue;

            foreach (var assignment in CartesianProduct(possibleAssignments))
            {
                int[] supplyUsage = new int[nSupply];
                bool valid = true;
                foreach (int supplyIdx in assignment)
                {
                    if (supplyIdx < 0) continue;
                    supplyUsage[supplyIdx]++;
                    if (supplyUsage[supplyIdx] > 1) { valid = false; break; }
                }
                if (!valid) continue;

                double totalScore = 0;
                var pairs = new List<MatchPair>();
                var unmatched = new List<Element>();

                for (int d = 0; d < nDemand; d++)
                {
                    int s = assignment[d];
                    if (s >= 0)
                    {
                        double score = weights[d, s];
                        totalScore += score;
                        pairs.Add(new MatchPair(demandElements[d], supplyElements[s], score));
                    }
                    else
                    {
                        unmatched.Add(demandElements[d]);
                    }
                }

                if (totalScore < bestScore)
                {
                    bestScore = totalScore;
                    var usedSupplyIds = new HashSet<int>(assignment.Where(x => x >= 0));
                    var unusedSupply = supplyElements
                        .Where((_, idx) => !usedSupplyIds.Contains(idx))
                        .ToList();

                    bestResult = new MatchResult
                    {
                        Pairs = pairs,
                        UnmatchedDemand = unmatched,
                        UnmatchedSupply = unusedSupply,
                        TotalScore = totalScore,
                        Method = "BruteForce"
                    };
                }
            }

            if (bestResult == null)
            {
                bestResult = new MatchResult
                {
                    UnmatchedDemand = new List<Element>(demandElements),
                    UnmatchedSupply = new List<Element>(supplyElements),
                    TotalScore = 0,
                    Method = "BruteForce"
                };
            }

            return bestResult;
        }

        /// <summary>
        /// Build incidence matrix: incidence[d,s] = true if supply s can serve demand d.
        /// For beams: supply.Length >= demand.Length, supply.Area >= demand.Area, supply.Iy >= demand.Iy
        /// For plates: supply.Thickness >= demand.Thickness (section level)
        /// Elements must be of the same type to match.
        /// </summary>
        public static bool[,] EvaluateIncidence(List<Element> demand, List<Element> supply)
        {
            int nD = demand.Count;
            int nS = supply.Count;
            bool[,] incidence = new bool[nD, nS];

            for (int d = 0; d < nD; d++)
            {
                for (int s = 0; s < nS; s++)
                {
                    incidence[d, s] = CheckConstraints(demand[d], supply[s]);
                }
            }

            return incidence;
        }

        /// <summary>
        /// Check if a supply element can satisfy a demand element based on type-specific constraints.
        /// </summary>
        public static bool CheckConstraints(Element demand, Element supply)
        {
            if (demand.GetType() != supply.GetType())
                return false;

            if (demand is Beam demandBeam && supply is Beam supplyBeam)
            {
                if (demandBeam.Section == null || supplyBeam.Section == null)
                    return false;

                return supplyBeam.Length >= demandBeam.Length
                    && supplyBeam.Section.Area >= demandBeam.Section.Area
                    && supplyBeam.Section.Iy >= demandBeam.Section.Iy
                    && supplyBeam.Section.Iz >= demandBeam.Section.Iz;
            }

            if (demand is Plate demandPlate && supply is Plate supplyPlate)
            {
                if (demandPlate.Section == null || supplyPlate.Section == null)
                    return false;

                bool thicknessOk = supplyPlate.Section.Thickness >= demandPlate.Section.Thickness;

                // If both plates have a length axis (material list panels), check length too
                bool supplyHasLength = supplyPlate.AxisLine.IsValid && supplyPlate.AxisLine.Length > 0;
                bool demandHasLength = demandPlate.AxisLine.IsValid && demandPlate.AxisLine.Length > 0;
                bool lengthOk = (!supplyHasLength || !demandHasLength)
                    || supplyPlate.AxisLine.Length >= demandPlate.AxisLine.Length;

                return thicknessOk && lengthOk;
            }

            return false;
        }

        /// <summary>
        /// Weight/score matrix: lower is better. Uses volume-based LCA proxy.
        /// For beams: score = supply.Length * supply.Section.Area (volume proxy for reuse benefit).
        /// Uses demand length (element is cut to demand length) following structuralCircle convention.
        /// </summary>
        public static double[,] EvaluateWeights(List<Element> demand, List<Element> supply)
        {
            int nD = demand.Count;
            int nS = supply.Count;
            double[,] weights = new double[nD, nS];

            for (int d = 0; d < nD; d++)
            {
                for (int s = 0; s < nS; s++)
                {
                    weights[d, s] = CalculateWeight(demand[d], supply[s]);
                }
            }

            return weights;
        }

        static double CalculateWeight(Element demand, Element supply)
        {
            if (demand is Beam demandBeam && supply is Beam supplyBeam)
            {
                double usedLength = demandBeam.Length;
                double area = supplyBeam.Section?.Area ?? 0;
                return usedLength * area;
            }

            if (demand is Plate demandPl && supply is Plate supplyPl)
            {
                double thickness = supplyPl.Section?.Thickness ?? double.MaxValue;
                // Volume proxy: use demand length (cut-to-fit), supply thickness
                bool supplyHasLength = supplyPl.AxisLine.IsValid && supplyPl.AxisLine.Length > 0;
                bool demandHasLength = demandPl.AxisLine.IsValid && demandPl.AxisLine.Length > 0;
                double usedLength = demandHasLength ? demandPl.AxisLine.Length
                                  : (supplyHasLength ? supplyPl.AxisLine.Length : 1.0);
                return usedLength * thickness;
            }

            return double.MaxValue;
        }

        /// <summary>
        /// For each demand element, extract the list of feasible supply indices.
        /// Returns List[demandIdx] -> List of feasible supplyIdx (plus -1 for "unmatched").
        /// </summary>
        static List<List<int>> ExtractBrutePossibilities(bool[,] incidence, int nDemand, int nSupply)
        {
            var possibilities = new List<List<int>>();

            for (int d = 0; d < nDemand; d++)
            {
                var feasible = new List<int>();
                for (int s = 0; s < nSupply; s++)
                {
                    if (incidence[d, s])
                        feasible.Add(s);
                }
                feasible.Add(-1);
                possibilities.Add(feasible);
            }

            return possibilities;
        }

        /// <summary>
        /// Cartesian product of lists (equivalent to itertools.product in Python).
        /// </summary>
        static IEnumerable<int[]> CartesianProduct(List<List<int>> lists)
        {
            if (lists.Count == 0)
            {
                yield return Array.Empty<int>();
                yield break;
            }

            int[] indices = new int[lists.Count];
            int[] maxes = lists.Select(l => l.Count).ToArray();

            while (true)
            {
                int[] result = new int[lists.Count];
                for (int i = 0; i < lists.Count; i++)
                    result[i] = lists[i][indices[i]];
                yield return result;

                int pos = lists.Count - 1;
                while (pos >= 0)
                {
                    indices[pos]++;
                    if (indices[pos] < maxes[pos]) break;
                    indices[pos] = 0;
                    pos--;
                }
                if (pos < 0) break;
            }
        }
    }
}
