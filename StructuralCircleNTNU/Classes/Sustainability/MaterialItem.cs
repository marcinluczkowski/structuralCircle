namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// One material line in a <see cref="DesignConcept"/>. Quantity can be given in kg or m³
    /// (picked with <see cref="Unit"/>); emission factor is per that same unit.
    /// </summary>
    public class MaterialItem
    {
        public string Name { get; set; } = "";
        public string MaterialType { get; set; } = "";

        public double QuantityM3 { get; set; }
        public double QuantityKg { get; set; }
        public double DensityKgM3 { get; set; }

        /// <summary>Share of this item that is reused (0–1).</summary>
        public double ReuseShare { get; set; }

        /// <summary>Production emission factor (kgCO2e per <see cref="Unit"/>).</summary>
        public double EmissionFactorA1A3 { get; set; }

        /// <summary>"kg" or "m3".</summary>
        public string Unit { get; set; } = "kg";

        public double TransportDistanceKm { get; set; }
        public double TransportFactorKgCO2ePerTkm { get; set; } = 0.1;

        /// <summary>Fraction of material lost as waste on site (0–1).</summary>
        public double WasteFactor { get; set; } = 0.05;

        public MaterialItem() { }

        public override string ToString()
        {
            string qty = Unit == "m3" ? $"{QuantityM3:F3} m³" : $"{QuantityKg:F1} kg";
            return $"Material '{Name}' ({MaterialType}, {qty}, reuse={ReuseShare * 100:F0}%)";
        }
    }
}
