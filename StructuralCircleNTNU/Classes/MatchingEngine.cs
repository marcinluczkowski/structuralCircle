using System;
using System.Collections.Generic;
using System.Linq;

namespace StructuralCircleNTNU.Classes
{
    public static class MatchingEngine
    {
        // ── public entry points ───────────────────────────────────────────

        /// <summary>
        /// Exhaustive brute-force: enumerate every valid one-to-one assignment and keep the
        /// one with minimum total waste volume.
        /// Exact and optimal but O((nS+1)^nD) — only practical for small problems (≤ 8–10 each).
        /// </summary>
        public static MatchResult BruteForceMatch(DemandBank demand, SupplyBank supply)
        {
            var dem = demand.Elements;
            var sup = supply.Elements;
            int nD = dem.Count, nS = sup.Count;

            bool[,] inc = EvaluateIncidence(dem, sup);
            double[,] w  = EvaluateWeights(dem, sup);

            var possibilities = ExtractBrutePossibilities(inc, nD, nS);
            MatchResult best  = null;
            double bestScore  = double.MaxValue;

            foreach (var assignment in CartesianProduct(possibilities))
            {
                int[] usage = new int[nS];
                bool valid  = true;
                foreach (int s in assignment)
                {
                    if (s < 0) continue;
                    if (++usage[s] > 1) { valid = false; break; }
                }
                if (!valid) continue;

                double total = 0;
                var pairs    = new List<MatchPair>();
                var unmatched = new List<Element>();

                for (int d = 0; d < nD; d++)
                {
                    int s = assignment[d];
                    if (s >= 0) { total += w[d, s]; pairs.Add(new MatchPair(dem[d], sup[s], w[d, s])); }
                    else          unmatched.Add(dem[d]);
                }

                if (total < bestScore)
                {
                    bestScore = total;
                    var usedSet = new HashSet<int>(assignment.Where(x => x >= 0));
                    best = new MatchResult
                    {
                        Pairs           = pairs,
                        UnmatchedDemand = unmatched,
                        UnmatchedSupply = sup.Where((_, i) => !usedSet.Contains(i)).ToList(),
                        TotalScore      = total,
                        Method          = "BruteForce"
                    };
                }
            }

            return best ?? new MatchResult
            {
                UnmatchedDemand = new List<Element>(dem),
                UnmatchedSupply = new List<Element>(sup),
                TotalScore      = 0,
                Method          = "BruteForce"
            };
        }

        /// <summary>
        /// Greedy heuristic: sort demand by volume descending (largest / hardest first), then for
        /// each demand greedily pick the feasible supply element with minimum waste volume.
        /// O(nD × nS × log nD). Not globally optimal but fast for large banks.
        /// Mirrors greedy_single from the structuralCircle Python repo.
        /// </summary>
        public static MatchResult GreedyMatch(DemandBank demand, SupplyBank supply)
        {
            // Sort demand descending by volume so the elements hardest to match are handled first.
            var dem = demand.Elements
                            .Select((e, i) => (elem: e, origIdx: i))
                            .OrderByDescending(x => ElementVolume(x.elem))
                            .ToList();
            var sup = supply.Elements;

            var available  = Enumerable.Range(0, sup.Count).ToList();
            var pairs      = new List<MatchPair>();
            var unmatched  = new List<Element>();
            var usedIdx    = new HashSet<int>();

            foreach (var (d, _) in dem)
            {
                int    bestS     = -1;
                double bestWaste = double.MaxValue;

                foreach (int si in available)
                {
                    if (!CheckConstraints(d, sup[si])) continue;
                    double waste = CalculateWaste(d, sup[si]);
                    if (waste < bestWaste) { bestWaste = waste; bestS = si; }
                }

                if (bestS >= 0)
                {
                    pairs.Add(new MatchPair(d, sup[bestS], bestWaste));
                    available.Remove(bestS);
                    usedIdx.Add(bestS);
                }
                else
                {
                    unmatched.Add(d);
                }
            }

            return new MatchResult
            {
                Pairs           = pairs,
                UnmatchedDemand = unmatched,
                UnmatchedSupply = sup.Where((_, i) => !usedIdx.Contains(i)).ToList(),
                TotalScore      = pairs.Sum(p => p.Score),
                Method          = "Greedy"
            };
        }

        /// <summary>
        /// Hungarian / bipartite optimal matching: finds the globally optimal one-to-one assignment
        /// minimising total waste volume. O(n³) where n = max(nD, nS).
        /// Equivalent to the bipartite method in the structuralCircle Python repo.
        ///
        /// Cost model:
        ///   feasible pair  → actual waste volume (m³, ≥ 0)
        ///   infeasible pair → 1e12  (never chosen)
        ///   demand left unmatched → 1e6 per demand  (prefer any feasible match over leaving unmatched)
        /// </summary>
        public static MatchResult HungarianMatch(DemandBank demand, SupplyBank supply)
        {
            var dem = demand.Elements;
            var sup = supply.Elements;
            int nD = dem.Count, nS = sup.Count;

            const double INFEASIBLE = 1e12; // pair violates constraints
            const double UNMATCHED  = 1e6;  // cost of leaving a demand element unmatched

            // Rectangular cost matrix: nD rows × (nS + nD) columns.
            // Columns 0..nS-1   = real supply elements.
            // Column  nS + d    = "no match" slot exclusive to demand d (cost = UNMATCHED).
            // All other dummy slots for demand d get INFEASIBLE (prevents cross-assignment).
            int nCols = nS + nD;
            double[,] cost = new double[nD, nCols];

            for (int d = 0; d < nD; d++)
            {
                for (int s = 0; s < nS; s++)
                    cost[d, s] = CheckConstraints(dem[d], sup[s])
                        ? CalculateWaste(dem[d], sup[s])
                        : INFEASIBLE;

                for (int k = 0; k < nD; k++)
                    cost[d, nS + k] = (k == d) ? UNMATCHED : INFEASIBLE;
            }

            int[] assignment = RunHungarianRect(cost, nD, nCols);

            var pairs     = new List<MatchPair>();
            var unmatched = new List<Element>();
            var usedSup   = new HashSet<int>();

            for (int d = 0; d < nD; d++)
            {
                int col = assignment[d];
                if (col < nS && cost[d, col] < INFEASIBLE / 2)
                {
                    double waste = CalculateWaste(dem[d], sup[col]);
                    pairs.Add(new MatchPair(dem[d], sup[col], waste));
                    usedSup.Add(col);
                }
                else
                {
                    unmatched.Add(dem[d]);
                }
            }

            return new MatchResult
            {
                Pairs           = pairs,
                UnmatchedDemand = unmatched,
                UnmatchedSupply = sup.Where((_, i) => !usedSup.Contains(i)).ToList(),
                TotalScore      = pairs.Sum(p => p.Score),
                Method          = "Hungarian"
            };
        }

        // ── constraint / weight helpers (public for testing) ─────────────

        public static bool[,] EvaluateIncidence(List<Element> demand, List<Element> supply)
        {
            int nD = demand.Count, nS = supply.Count;
            bool[,] inc = new bool[nD, nS];
            for (int d = 0; d < nD; d++)
                for (int s = 0; s < nS; s++)
                    inc[d, s] = CheckConstraints(demand[d], supply[s]);
            return inc;
        }

        /// <summary>Waste-volume matrix (m³): lower is better.</summary>
        public static double[,] EvaluateWeights(List<Element> demand, List<Element> supply)
        {
            int nD = demand.Count, nS = supply.Count;
            double[,] w = new double[nD, nS];
            for (int d = 0; d < nD; d++)
                for (int s = 0; s < nS; s++)
                    w[d, s] = CalculateWaste(demand[d], supply[s]);
            return w;
        }

        /// <summary>
        /// Returns true if supply element structurally satisfies the demand element.
        /// Beams: supply ≥ demand in Length, section Area, Iy, Iz.
        /// Plates: supply ≥ demand in Thickness (and Length when both are defined).
        /// </summary>
        public static bool CheckConstraints(Element demand, Element supply)
        {
            if (demand.GetType() != supply.GetType()) return false;

            if (demand is Beam dB && supply is Beam sB)
            {
                if (dB.Section == null || sB.Section == null) return false;
                return sB.Length          >= dB.Length
                    && sB.Section.Area    >= dB.Section.Area
                    && sB.Section.Iy      >= dB.Section.Iy
                    && sB.Section.Iz      >= dB.Section.Iz;
            }

            if (demand is Plate dP && supply is Plate sP)
            {
                if (dP.Section == null || sP.Section == null) return false;
                bool tOk = sP.Section.Thickness >= dP.Section.Thickness;
                bool supL = sP.AxisLine.IsValid && sP.AxisLine.Length > 0;
                bool demL = dP.AxisLine.IsValid && dP.AxisLine.Length > 0;
                bool lOk  = (!supL || !demL) || sP.AxisLine.Length >= dP.AxisLine.Length;
                return tOk && lOk;
            }

            return false;
        }

        /// <summary>
        /// Waste volume (m³) = supply_volume − demand_volume.
        /// Beams: L × W × H; Plates: L × W × T.
        /// </summary>
        public static double CalculateWaste(Element demand, Element supply)
        {
            if (demand is Beam dB && supply is Beam sB)
            {
                double sv = sB.Length * (sB.Section?.Width ?? 0) * (sB.Section?.Height ?? 0);
                double dv = dB.Length * (dB.Section?.Width ?? 0) * (dB.Section?.Height ?? 0);
                return Math.Max(0, sv - dv);
            }

            if (demand is Plate dP && supply is Plate sP)
            {
                bool supL = sP.AxisLine.IsValid && sP.AxisLine.Length > 0;
                bool demL = dP.AxisLine.IsValid && dP.AxisLine.Length > 0;
                double sl = supL ? sP.AxisLine.Length : 0;
                double dl = demL ? dP.AxisLine.Length : 0;
                double sv = sl * (sP.Section?.Width ?? 0) * (sP.Section?.Thickness ?? 0);
                double dv = dl * (dP.Section?.Width ?? 0) * (dP.Section?.Thickness ?? 0);
                return Math.Max(0, sv - dv);
            }

            return double.MaxValue;
        }

        // ── Hungarian internals ───────────────────────────────────────────

        /// <summary>
        /// Standard O(n²·m) Kuhn-Munkres algorithm for rectangular matrices (nRows ≤ nCols).
        /// Returns assignment[row] = column (0-indexed) for minimum total cost.
        /// All costs must be finite (use large sentinels for infeasible / forbidden cells).
        /// </summary>
        static int[] RunHungarianRect(double[,] cost, int nRows, int nCols)
        {
            // 1-indexed internally to match the classic formulation.
            double[] u   = new double[nRows + 1]; // row potentials
            double[] v   = new double[nCols + 1]; // column potentials
            int[]  rowOf = new int[nCols + 1];    // rowOf[j] = row assigned to column j (0 = free)
            int[]  prev  = new int[nCols + 1];    // augmenting-path predecessor

            for (int i = 1; i <= nRows; i++)
            {
                rowOf[0] = i;
                int j0 = 0;

                double[] minD   = new double[nCols + 1];
                bool[]   inPath = new bool[nCols + 1];
                for (int j = 0; j <= nCols; j++) minD[j] = double.MaxValue / 2;

                // Dijkstra-like shortest-path step.
                do
                {
                    inPath[j0] = true;
                    int    i0    = rowOf[j0];
                    double delta = double.MaxValue / 2;
                    int    j1    = -1;

                    for (int j = 1; j <= nCols; j++)
                    {
                        if (inPath[j]) continue;
                        double reduced = cost[i0 - 1, j - 1] - u[i0] - v[j];
                        if (reduced < minD[j]) { minD[j] = reduced; prev[j] = j0; }
                        if (minD[j] < delta)   { delta   = minD[j]; j1      = j; }
                    }

                    // Update potentials.
                    for (int j = 0; j <= nCols; j++)
                    {
                        if (inPath[j]) { u[rowOf[j]] += delta; v[j] -= delta; }
                        else             minD[j]      -= delta;
                    }

                    j0 = j1;
                } while (rowOf[j0] != 0);

                // Augment along the path.
                do
                {
                    int j1 = prev[j0];
                    rowOf[j0] = rowOf[j1];
                    j0 = j1;
                } while (j0 != 0);
            }

            // Convert to 0-indexed result array.
            int[] ans = new int[nRows];
            for (int j = 1; j <= nCols; j++)
                if (rowOf[j] >= 1 && rowOf[j] <= nRows)
                    ans[rowOf[j] - 1] = j - 1;
            return ans;
        }

        // ── brute-force helpers ───────────────────────────────────────────

        static List<List<int>> ExtractBrutePossibilities(bool[,] inc, int nD, int nS)
        {
            var out_ = new List<List<int>>();
            for (int d = 0; d < nD; d++)
            {
                var f = new List<int>();
                for (int s = 0; s < nS; s++)
                    if (inc[d, s]) f.Add(s);
                f.Add(-1); // "leave unmatched" option
                out_.Add(f);
            }
            return out_;
        }

        static IEnumerable<int[]> CartesianProduct(List<List<int>> lists)
        {
            if (lists.Count == 0) { yield return Array.Empty<int>(); yield break; }

            int[] idx  = new int[lists.Count];
            int[] maxs = lists.Select(l => l.Count).ToArray();

            while (true)
            {
                int[] r = new int[lists.Count];
                for (int i = 0; i < lists.Count; i++) r[i] = lists[i][idx[i]];
                yield return r;

                int pos = lists.Count - 1;
                while (pos >= 0)
                {
                    if (++idx[pos] < maxs[pos]) break;
                    idx[pos--] = 0;
                }
                if (pos < 0) break;
            }
        }

        // ── utility ───────────────────────────────────────────────────────

        static double ElementVolume(Element e)
        {
            if (e is Beam  b) return b.Length * (b.Section?.Width ?? 0) * (b.Section?.Height    ?? 0);
            if (e is Plate p) return p.Length * (p.Section?.Width ?? 0) * (p.Section?.Thickness ?? 0);
            return 0;
        }
    }
}
