using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace StructuralCircleNTNU.Classes
{
    public class MatchPair
    {
        public Element Demand { get; set; }
        public Element Supply { get; set; }
        public double Score { get; set; }

        public MatchPair(Element demand, Element supply, double score)
        {
            Demand = demand;
            Supply = supply;
            Score = score;
        }

        public override string ToString()
        {
            return $"{Demand.Name} -> {Supply.Name} (score: {Score:F4})";
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
