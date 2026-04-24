namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// Connection line item. Supports both plain steel connectors and interlocking timber
    /// connections (CNC-milled joints): both kinds can be combined in one hybrid connection.
    /// </summary>
    public class ConnectionItem
    {
        public string Name { get; set; } = "";
        public int Count { get; set; } = 1;

        public double SteelMassKg { get; set; }
        public double SteelEmissionFactor { get; set; } = 1.5;

        /// <summary>Extra timber needed per single interlocking connection (kg).</summary>
        public double ExtraTimberKg { get; set; }
        public double TimberEmissionFactor { get; set; } = 0.35;

        public double CncEnergyKWh { get; set; }
        public double ElectricityEmissionFactor { get; set; } = 0.1;

        public double ToolWearCO2e { get; set; }

        /// <summary>Fraction of timber lost to CNC yield (0–1).</summary>
        public double YieldLoss { get; set; } = 0.0;

        public ConnectionItem() { }

        public override string ToString()
        {
            return $"Connection '{Name}' (×{Count}, steel={SteelMassKg:F2}kg, timber+={ExtraTimberKg:F2}kg)";
        }
    }
}
