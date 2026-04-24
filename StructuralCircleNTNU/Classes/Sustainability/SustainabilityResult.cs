using System.Text;

namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// Modular LCA result. Module D is stored separately (not added to the main totals automatically)
    /// but exposed through <see cref="OptionalTotalWithD"/>.
    /// </summary>
    public class SustainabilityResult
    {
        public string ConceptName { get; set; } = "";
        public string AllocationMethod { get; set; } = "";

        public double A1A3 { get; set; }
        public double A4 { get; set; }
        public double A5 { get; set; }
        public double C1C4 { get; set; }
        public double ModuleD { get; set; }

        public double TotalA1A5 { get; set; }
        public double TotalA1C4 { get; set; }
        public double OptionalTotalWithD { get; set; }

        public double CO2PerM2 { get; set; }
        public double CO2PerServiceYear { get; set; }

        public double ReuseRatio { get; set; }
        public double UtilisationRatio { get; set; }
        public double ReclamationPotential { get; set; }

        /// <summary>"X / Y" — number of connections where interlocking beats plain steel.</summary>
        public string InterlockingBreakEven { get; set; } = "";

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Sustainability '{ConceptName}' [{AllocationMethod}]");
            sb.AppendLine($"  A1–A3:  {A1A3,12:F2} kgCO2e");
            sb.AppendLine($"  A4:     {A4,12:F2} kgCO2e");
            sb.AppendLine($"  A5:     {A5,12:F2} kgCO2e");
            sb.AppendLine($"  C1–C4:  {C1C4,12:F2} kgCO2e");
            sb.AppendLine($"  Module D (reported separately): {ModuleD:F2} kgCO2e");
            sb.AppendLine($"  Total A1–A5:  {TotalA1A5,12:F2} kgCO2e");
            sb.AppendLine($"  Total A1–C4:  {TotalA1C4,12:F2} kgCO2e");
            sb.AppendLine($"  Total (+ D):  {OptionalTotalWithD,12:F2} kgCO2e");
            sb.AppendLine($"  CO2/m²:       {CO2PerM2,12:F2} kgCO2e/m²");
            sb.AppendLine($"  CO2/year:     {CO2PerServiceYear,12:F2} kgCO2e/yr");
            sb.AppendLine($"  Reuse ratio:           {ReuseRatio * 100:F1} %");
            sb.AppendLine($"  Utilisation ratio:     {UtilisationRatio * 100:F1} %");
            sb.AppendLine($"  Reclamation potential: {ReclamationPotential * 100:F1} %");
            if (!string.IsNullOrEmpty(InterlockingBreakEven))
                sb.AppendLine($"  Interlocking beats steel: {InterlockingBreakEven}");
            return sb.ToString();
        }
    }
}
