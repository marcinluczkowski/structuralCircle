using System.Collections.Generic;

namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// One design alternative: all material lines + connections + service-life parameters needed
    /// to run a module-wise LCA (A1–A5, C1–C4, D).
    /// </summary>
    public class DesignConcept
    {
        public string Name { get; set; } = "";

        public List<MaterialItem> Materials { get; set; } = new List<MaterialItem>();
        public List<ConnectionItem> Connections { get; set; } = new List<ConnectionItem>();

        public double ReferenceStudyPeriodYears { get; set; } = 50;
        public double FunctionalUnitAreaM2 { get; set; } = 1;

        /// <summary>0–1. Probability that the building's materials are actually reused after its life.</summary>
        public double FutureReuseProbability { get; set; } = 0.5;

        /// <summary>Efficiency of avoided production in Module D (0–1).</summary>
        public double SubstitutionFactor { get; set; } = 1.0;

        /// <summary>0–1. Average structural utilisation ratio across members (for reporting).</summary>
        public double UtilisationRatio { get; set; } = 1.0;

        public DesignConcept() { }

        public override string ToString()
        {
            return $"DesignConcept '{Name}' (mats={Materials.Count}, conns={Connections.Count}, {ReferenceStudyPeriodYears:F0} yr, {FunctionalUnitAreaM2:F1} m²)";
        }
    }
}
