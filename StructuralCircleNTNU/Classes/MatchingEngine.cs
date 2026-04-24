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
        /// <summary>Default cap on Cartesian combinations before refusing brute force (avoids freezing Rhino).</summary>
        public const long DefaultMaxBruteCombinations = 2_000_000;

        /// <summary>
        /// Greedy matching: for each demand in list order, assign the feasible unused supply with lowest weight.
        /// O(demand × supply); safe for large banks.
        /// </summary>
        public static MatchResult GreedyMatch(DemandBank demand, SupplyBank supply)
        {
            var demandElements = demand.Elements;
            var supplyElements = supply.Elements;
            int nDemand = demandElements.Count;
            int nSupply = supplyElements.Count;

            if (nDemand == 0)
            {
                return new MatchResult
                {
                    UnmatchedSupply = new List<Element>(supplyElements),
                    TotalScore = 0,
                    Method = "Greedy"
                };
            }

            bool[,] incidence = EvaluateIncidence(demandElements, supplyElements);
            double[,] weights = EvaluateWeights(demandElements, supplyElements);

            var usedSupply = new bool[nSupply];
            var pairs = new List<MatchPair>();
            var unmatchedDemand = new List<Element>();

            for (int d = 0; d < nDemand; d++)
            {
                int bestS = -1;
                double bestW = double.MaxValue;
                for (int s = 0; s < nSupply; s++)
                {
                    if (usedSupply[s] || !incidence[d, s]) continue;
                    double w = weights[d, s];
                    if (w < bestW)
                    {
                        bestW = w;
                        bestS = s;
                    }
                }

                if (bestS >= 0)
                {
                    usedSupply[bestS] = true;
                    pairs.Add(new MatchPair(demandElements[d], supplyElements[bestS], bestW));
                }
                else
                    unmatchedDemand.Add(demandElements[d]);
            }

            var unmatchedSupply = supplyElements
                .Where((_, idx) => !usedSupply[idx])
                .ToList();

            return new MatchResult
            {
                Pairs = pairs,
                UnmatchedDemand = unmatchedDemand,
                UnmatchedSupply = unmatchedSupply,
                TotalScore = pairs.Sum(p => p.Score),
                Method = "Greedy"
            };
        }

        /// <summary>
        /// Brute force on a subset of demand (see <paramref name="demandSubsetFraction"/> / <paramref name="maxDemandElements"/>).
        /// Remaining demand rows are left unmatched (not part of the search).
        /// </summary>
        public static MatchResult BruteForceMatch(DemandBank demand, SupplyBank supply)
            => BruteForceMatch(demand, supply, demandSubsetFraction: 0.1, maxDemandElements: 0, maxCombinations: DefaultMaxBruteCombinations);

        /// <param name="demandSubsetFraction">Fraction of demand count in brute search when <paramref name="maxDemandElements"/> is 0 (clamped to (0,1]).</param>
        /// <param name="maxDemandElements">If &gt; 0, use this many demand elements (from the start of the list) instead of the fraction.</param>
        /// <param name="maxCombinations">Abort before enumerating if the Cartesian product exceeds this.</param>
        public static MatchResult BruteForceMatch(
            DemandBank demand,
            SupplyBank supply,
            double demandSubsetFraction,
            int maxDemandElements,
            long maxCombinations = DefaultMaxBruteCombinations)
        {
            var demandElements = demand.Elements;
            var supplyElements = supply.Elements;

            int nDemand = demandElements.Count;
            int nSupply = supplyElements.Count;

            if (nDemand == 0)
            {
                return new MatchResult
                {
                    UnmatchedSupply = new List<Element>(supplyElements),
                    TotalScore = 0,
                    Method = "BruteForce"
                };
            }

            int kBrute = ResolveBruteDemandCount(nDemand, demandSubsetFraction, maxDemandElements);
            var bruteDemandIndices = Enumerable.Range(0, kBrute).ToList();
            var bruteDemands = bruteDemandIndices.Select(i => demandElements[i]).ToList();

            bool[,] incidenceFull = EvaluateIncidence(demandElements, supplyElements);
            double[,] weightsFull = EvaluateWeights(demandElements, supplyElements);

            bool[,] incidence = SubMatrix(incidenceFull, bruteDemandIndices, nSupply);
            double[,] weights = SubMatrix(weightsFull, bruteDemandIndices, nSupply);

            var possibleAssignments = ExtractBrutePossibilities(incidence, kBrute, nSupply);

            long est = EstimateCartesianSize(possibleAssignments, maxCombinations);
            if (est > maxCombinations)
            {
                var note =
                    $"Brute force skipped: about {est:N0} combinations (limit {maxCombinations:N0}). " +
                    "Lower DemandSubset, MaxDemand, or use Greedy (mode 0).";
                return AbortedBruteResult(demandElements, supplyElements, kBrute, note);
            }

            MatchResult bestSubset = null;
            double bestScore = double.MaxValue;

            foreach (var assignment in CartesianProduct(possibleAssignments))
            {
                var supplyUsage = new int[nSupply];
                bool valid = true;
                foreach (int supplyIdx in assignment)
                {
                    if (supplyIdx < 0) continue;
                    supplyUsage[supplyIdx]++;
                    if (supplyUsage[supplyIdx] > 1)
                    {
                        valid = false;
                        break;
                    }
                }

                if (!valid) continue;

                double totalScore = 0;
                var pairsSubset = new List<MatchPair>();
                var unmatchedInBrute = new List<Element>();

                for (int di = 0; di < kBrute; di++)
                {
                    int globalD = bruteDemandIndices[di];
                    int s = assignment[di];
                    if (s >= 0)
                    {
                        double score = weights[di, s];
                        totalScore += score;
                        pairsSubset.Add(new MatchPair(demandElements[globalD], supplyElements[s], score));
                    }
                    else
                        unmatchedInBrute.Add(demandElements[globalD]);
                }

                if (totalScore < bestScore)
                {
                    bestScore = totalScore;
                    var usedSupplyIds = new HashSet<int>(assignment.Where(x => x >= 0));
                    var unusedSupply = supplyElements
                        .Where((_, idx) => !usedSupplyIds.Contains(idx))
                        .ToList();

                    var tailUnmatchedDemand = demandElements
                        .Skip(kBrute)
                        .ToList();
                    unmatchedInBrute.AddRange(tailUnmatchedDemand);

                    bestSubset = new MatchResult
                    {
                        Pairs = pairsSubset,
                        UnmatchedDemand = unmatchedInBrute,
                        UnmatchedSupply = unusedSupply,
                        TotalScore = totalScore,
                        Method = "BruteForce",
                        Note = kBrute < nDemand
                            ? $"Optimized first {kBrute} of {nDemand} demand elements; remainder listed as unmatched."
                            : null
                    };
                }
            }

            if (bestSubset == null)
            {
                return new MatchResult
                {
                    UnmatchedDemand = new List<Element>(demandElements),
                    UnmatchedSupply = new List<Element>(supplyElements),
                    TotalScore = 0,
                    Method = "BruteForce"
                };
            }

            return bestSubset;
        }

        static MatchResult AbortedBruteResult(List<Element> demandElements, List<Element> supplyElements, int kBrute, string note)
        {
            var unmatched = new List<Element>(demandElements);
            return new MatchResult
            {
                Pairs = new List<MatchPair>(),
                UnmatchedDemand = unmatched,
                UnmatchedSupply = new List<Element>(supplyElements),
                TotalScore = 0,
                Method = "BruteForce",
                Note = note
            };
        }

        /// <summary>How many leading demand elements participate in brute force.</summary>
        public static int ResolveBruteDemandCount(int nDemand, double demandSubsetFraction, int maxDemandElements)
        {
            if (nDemand <= 0) return 0;
            int fromFrac;
            if (maxDemandElements > 0)
                fromFrac = Math.Min(maxDemandElements, nDemand);
            else
            {
                double f = demandSubsetFraction;
                if (f <= 0 || f > 1) f = 0.1;
                fromFrac = (int)Math.Ceiling(f * nDemand);
            }

            return Math.Max(1, Math.Min(fromFrac, nDemand));
        }

        static long EstimateCartesianSize(List<List<int>> possibilities, long cap)
        {
            double p = 1;
            foreach (var list in possibilities)
            {
                if (list.Count == 0) return 0;
                p *= list.Count;
                if (p > cap || p > long.MaxValue) return long.MaxValue;
            }

            return (long)Math.Min(p, long.MaxValue);
        }

        static bool[,] SubMatrix(bool[,] full, List<int> rowIndices, int nCols)
        {
            int nR = rowIndices.Count;
            var m = new bool[nR, nCols];
            for (int i = 0; i < nR; i++)
            {
                int r = rowIndices[i];
                for (int c = 0; c < nCols; c++)
                    m[i, c] = full[r, c];
            }

            return m;
        }

        static double[,] SubMatrix(double[,] full, List<int> rowIndices, int nCols)
        {
            int nR = rowIndices.Count;
            var m = new double[nR, nCols];
            for (int i = 0; i < nR; i++)
            {
                int r = rowIndices[i];
                for (int c = 0; c < nCols; c++)
                    m[i, c] = full[r, c];
            }

            return m;
        }

        /// <summary>
        /// MLP-inspired matching: feature-based pair scoring (normalised excess ratios through sigmoid
        /// activations — mimicking the output layer of a trained MLP) combined with the Hungarian
        /// algorithm for globally optimal assignment.
        /// Reference: Luczkowski et al., "Matching reclaimed structural members", IOP Environ. Res.
        /// Infrastruct. Sustain. 3 (2023).
        /// </summary>
        public static MatchResult MlpMatch(DemandBank demand, SupplyBank supply)
        {
            var demandElems = demand.Elements;
            var supplyElems = supply.Elements;
            int nD = demandElems.Count;
            int nS = supplyElems.Count;

            if (nD == 0)
            {
                return new MatchResult
                {
                    UnmatchedSupply = new List<Element>(supplyElems),
                    TotalScore = 0,
                    Method = "MLP"
                };
            }

            double[,] scoreMtx = new double[nD, nS];
            for (int d = 0; d < nD; d++)
                for (int s = 0; s < nS; s++)
                    scoreMtx[d, s] = MlpPairScore(demandElems[d], supplyElems[s]);

            int[] assignment = RunHungarian(scoreMtx, nD, nS);

            const double InfThreshold = 1e17;
            var pairs = new List<MatchPair>();
            var unmatchedDemand = new List<Element>();
            var usedSupply = new bool[nS];

            for (int d = 0; d < nD; d++)
            {
                int s = assignment[d];
                if (s >= 0 && scoreMtx[d, s] < InfThreshold)
                {
                    pairs.Add(new MatchPair(demandElems[d], supplyElems[s], scoreMtx[d, s]));
                    usedSupply[s] = true;
                }
                else
                    unmatchedDemand.Add(demandElems[d]);
            }

            var unmatchedSupply = supplyElems.Where((_, i) => !usedSupply[i]).ToList();

            return new MatchResult
            {
                Pairs = pairs,
                UnmatchedDemand = unmatchedDemand,
                UnmatchedSupply = unmatchedSupply,
                TotalScore = pairs.Sum(p => p.Score),
                Method = "MLP",
                Note = "Feature-based MLP scoring (sigmoid of normalised excess ratios) + Hungarian optimal assignment."
            };
        }

        /// <summary>
        /// Pair score for MLP mode.
        /// For feasible pairs: sum of sigmoid activations on the normalised excess in each
        /// structural property (Length, Area, Iy, Iz for beams; Thickness [+Length] for plates).
        /// Lower score = tighter fit = better match.
        /// Infeasible pairs return a large sentinel value.
        /// </summary>
        static double MlpPairScore(Element demand, Element supply)
        {
            const double Inf = 2e18;
            if (!CheckConstraints(demand, supply)) return Inf;

            if (demand is Beam dBeam && supply is Beam sBeam)
            {
                double dL  = Math.Max(dBeam.Length, 1e-9);
                double dA  = Math.Max(dBeam.Section?.Area ?? 1e-9, 1e-9);
                double dIy = Math.Max(dBeam.Section?.Iy   ?? 1e-9, 1e-9);
                double dIz = Math.Max(dBeam.Section?.Iz   ?? 1e-9, 1e-9);

                double eL  = (sBeam.Length              - dL)  / dL;
                double eA  = ((sBeam.Section?.Area  ?? 0) - dA)  / dA;
                double eIy = ((sBeam.Section?.Iy    ?? 0) - dIy) / dIy;
                double eIz = ((sBeam.Section?.Iz    ?? 0) - dIz) / dIz;

                return Sigmoid(eL) + Sigmoid(eA) + Sigmoid(eIy) + Sigmoid(eIz);
            }

            if (demand is Plate dPlate && supply is Plate sPlate)
            {
                double dT = Math.Max(dPlate.Section?.Thickness ?? 1e-9, 1e-9);
                double eT = ((sPlate.Section?.Thickness ?? 0) - dT) / dT;
                double score = Sigmoid(eT);

                double dL = dPlate.Length;
                double sL = sPlate.Length;
                if (dL > 1e-9 && sL > 1e-9)
                    score += Sigmoid((sL - dL) / dL);

                return score;
            }

            return Inf;
        }

        static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        /// <summary>
        /// Kuhn-Munkres / Hungarian algorithm, O(n³).
        /// Returns assignment[d] = s (0-based), or -1 if demand d is left unmatched.
        /// Cost matrix is padded to square with a large sentinel when nD ≠ nS.
        /// Assignments to the padded dummy columns are treated as "unmatched".
        /// </summary>
        static int[] RunHungarian(double[,] cost, int nD, int nS)
        {
            const double INF = 2e18;
            int n = Math.Max(nD, nS);

            double[] u   = new double[n + 1];
            double[] v   = new double[n + 1];
            int[]    p   = new int[n + 1];
            int[]    way = new int[n + 1];

            for (int i = 1; i <= n; i++)
            {
                p[0] = i;
                int j0 = 0;
                var minv = new double[n + 1];
                for (int k = 0; k <= n; k++) minv[k] = INF;
                var used = new bool[n + 1];

                do
                {
                    used[j0] = true;
                    int i0 = p[j0];
                    double delta = INF;
                    int j1 = 0;

                    for (int j = 1; j <= n; j++)
                    {
                        if (used[j]) continue;
                        int ri = i0 - 1, ci = j - 1;
                        double c = (ri < nD && ci < nS) ? cost[ri, ci] : INF;
                        double r = c - u[i0] - v[j];
                        if (r < minv[j]) { minv[j] = r; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }

                    for (int j = 0; j <= n; j++)
                    {
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else minv[j] -= delta;
                    }
                    j0 = j1;
                }
                while (p[j0] != 0);

                do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; }
                while (j0 != 0);
            }

            int[] assign = new int[nD];
            for (int d = 0; d < nD; d++) assign[d] = -1;

            for (int j = 1; j <= nS; j++)
            {
                int row = p[j];
                if (row < 1 || row > nD) continue;
                int d = row - 1;
                int s = j - 1;
                if (cost[d, s] < INF / 2)
                    assign[d] = s;
            }

            return assign;
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
