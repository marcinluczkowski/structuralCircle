namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Matching strategy for supply/demand banks (Grasshopper int input).
    /// 0 = Greedy                — fast one-pass heuristic.
    /// 1 = BruteForce            — exact enumeration on a limited demand subset.
    /// 2 = MILP                  — Mixed-Integer Linear Programming bipartite matching (Hungarian LP-optimum).
    /// 3 = GreedyPacking         — greedy matching + packing heuristic (multiple demand per supply).
    /// 4 = MilpPacking           — MILP-style packed matching (BFD + local-search improvement).
    /// </summary>
    public enum MatchingAlgorithmMode
    {
        Greedy = 0,
        BruteForce = 1,
        Milp = 2,
        GreedyPacking = 3,
        MilpPacking = 4
    }

    /// <summary>
    /// Packing geometry used by packed matching modes (3 and 4).
    /// </summary>
    public enum PackingMode
    {
        /// <summary>Automatic: 1D for beams, 2D shelf for plates, 3D BBox for mixed.</summary>
        Auto = 0,
        /// <summary>1D cutting stock along element length (beams).</summary>
        Length1D = 1,
        /// <summary>2D shelf packing on plate area (thickness must match).</summary>
        Area2D = 2,
        /// <summary>3D bounding-box volume packing (fallback for heterogeneous shapes).</summary>
        Bbox3D = 3,
        /// <summary>Use the actual Brep geometry (falls back to 3D BBox when not available).</summary>
        Brep = 4
    }
}
