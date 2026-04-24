using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    public class MatchPair
    {
        public Element Demand { get; set; }
        public Element Supply { get; set; }
        public double Score { get; set; }

        /// <summary>
        /// Optional placement of the demand element inside the supply (packing / cutting-stock).
        /// 1D cutting stock → a line along the member axis, start at the cut offset, end at cut offset + demand length.
        /// 2D shelf packing  → a line from (x0, y0, 0) to (x0 + demandWidth, y0 + demandHeight, 0).
        /// 3D BBox packing   → a line from the min corner to the max corner of the demand BBox inside supply.
        /// Line.Unset (default) when packing is not used (plain one-to-one match).
        /// </summary>
        public Line Placement { get; set; } = Line.Unset;

        /// <summary>True when <see cref="Placement"/> encodes a packing position.</summary>
        public bool HasPlacement => Placement.IsValid;

        public MatchPair(Element demand, Element supply, double score)
        {
            Demand = demand;
            Supply = supply;
            Score = score;
        }

        public override string ToString()
        {
            string place = HasPlacement
                ? $" @ [{Placement.FromX:F3}, {Placement.FromY:F3}, {Placement.FromZ:F3}]"
                : "";
            return $"{Demand.Name} -> {Supply.Name}{place} (score: {Score:F4})";
        }
    }

    public class MatchResult
    {
        public List<MatchPair> Pairs { get; set; }
        public List<Element> UnmatchedDemand { get; set; }
        public List<Element> UnmatchedSupply { get; set; }
        public double TotalScore { get; set; }
        public string Method { get; set; }

        /// <summary>Optional note when matching was skipped, aborted, or not implemented.</summary>
        public string Note { get; set; }

        public MatchResult()
        {
            Pairs = new List<MatchPair>();
            UnmatchedDemand = new List<Element>();
            UnmatchedSupply = new List<Element>();
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"MatchResult ({Method}): {Pairs.Count} pairs, score={TotalScore:F4}");
            if (!string.IsNullOrEmpty(Note))
                sb.AppendLine($"  Note: {Note}");
            sb.AppendLine($"  Unmatched demand: {UnmatchedDemand.Count}, Unmatched supply: {UnmatchedSupply.Count}");
            foreach (var pair in Pairs)
                sb.AppendLine($"  {pair}");
            return sb.ToString();
        }
    }
}
