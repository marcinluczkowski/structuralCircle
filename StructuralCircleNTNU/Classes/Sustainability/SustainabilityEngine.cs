using System.Collections.Generic;
using System.Linq;

namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// Modular module-wise LCA engine (A1–A5, C1–C4, D). Every sub-step is a separate public
    /// static function so the pipeline can be extended (Monte Carlo, detailed EPDs, finer C1–C4)
    /// without touching the rest.
    /// </summary>
    public static class SustainabilityEngine
    {
        // ── Allocation factors (reused share burden split) ───────────────────────
        public static double CutOffFactor(double reuseShare) => 1.0 - reuseShare;

        public static double PEF5050Factor(double reuseShare) => (1.0 - reuseShare) + 0.5 * reuseShare;

        public static double NCycleFactor(double reuseShare, double numberOfCycles)
        {
            double n = numberOfCycles > 0 ? numberOfCycles : 1.0;
            return (1.0 - reuseShare) + reuseShare / n;
        }

        public static double AllocationFactor(double reuseShare, AllocationMethod method, double numberOfCycles)
        {
            if (reuseShare <= 0) return 1.0;
            switch (method)
            {
                case AllocationMethod.CutOff: return CutOffFactor(reuseShare);
                case AllocationMethod.PEF50_50: return PEF5050Factor(reuseShare);
                case AllocationMethod.NCycle: return NCycleFactor(reuseShare, numberOfCycles);
                default: return 1.0;
            }
        }

        // ── Quantity helper ─────────────────────────────────────────────────────
        public static double GetQuantity(MaterialItem m)
        {
            if (m == null) return 0;
            if (m.Unit == "m3") return m.QuantityM3;
            return m.QuantityKg;
        }

        // ── Module-level calculations ───────────────────────────────────────────
        public static double CalculateA1A3(MaterialItem m, AllocationMethod method, double numberOfCycles)
        {
            if (m == null) return 0;
            double quantity = GetQuantity(m);
            double factor = AllocationFactor(m.ReuseShare, method, numberOfCycles);
            return quantity * m.EmissionFactorA1A3 * factor;
        }

        public static double CalculateA4(MaterialItem m)
        {
            if (m == null) return 0;
            double massT = m.QuantityKg / 1000.0;
            return massT * m.TransportDistanceKm * m.TransportFactorKgCO2ePerTkm;
        }

        public static double CalculateA5(MaterialItem m)
        {
            if (m == null) return 0;
            double wasted = GetQuantity(m) * m.WasteFactor;
            return wasted * m.EmissionFactorA1A3;
        }

        public static double CalculateSiteEnergy(double kWh, double electricityEF) => kWh * electricityEF;

        public static double CalculateC1C4(MaterialItem m, double endOfLifeFactor)
        {
            if (m == null) return 0;
            return m.QuantityKg * endOfLifeFactor;
        }

        // ── Connections ─────────────────────────────────────────────────────────
        public static double CalculateSteelConnection(ConnectionItem c)
        {
            if (c == null) return 0;
            return c.Count * c.SteelMassKg * c.SteelEmissionFactor;
        }

        public static double CalculateInterlockingConnection(ConnectionItem c)
        {
            if (c == null || c.ExtraTimberKg <= 0) return 0;

            double yieldMaterialKg = (1.0 - c.YieldLoss) > 1e-6
                ? c.ExtraTimberKg / (1.0 - c.YieldLoss) - c.ExtraTimberKg
                : 0;

            double materialCO2 = c.ExtraTimberKg * c.TimberEmissionFactor;
            double yieldCO2 = yieldMaterialKg * c.TimberEmissionFactor;
            double energyCO2 = c.CncEnergyKWh * c.ElectricityEmissionFactor;

            double perConnection = materialCO2 + yieldCO2 + energyCO2 + c.ToolWearCO2e;
            return c.Count * perConnection;
        }

        public static bool IsInterlockingBetterThanSteel(ConnectionItem c)
        {
            if (c == null || c.Count == 0 || c.ExtraTimberKg <= 0) return false;
            double steelPer = c.SteelMassKg * c.SteelEmissionFactor;
            double interPer = CalculateInterlockingConnection(c) / c.Count;
            return interPer < steelPer;
        }

        // ── Module D ────────────────────────────────────────────────────────────
        /// <summary>
        /// Module D (reported separately). Negative return value = climate benefit.
        /// </summary>
        public static double CalculateModuleD(
            double reusableMassKg,
            double substitutedEmissionFactor,
            double substitutionFactor,
            double futureReuseProbability,
            double processingCO2)
        {
            double avoided = reusableMassKg
                             * substitutedEmissionFactor
                             * substitutionFactor
                             * futureReuseProbability;
            return -avoided + processingCO2;
        }

        // ── Aggregate metrics ───────────────────────────────────────────────────
        public static double CalculateReuseRatio(IList<MaterialItem> materials)
        {
            double total = 0, reused = 0;
            foreach (var m in materials)
            {
                if (m == null) continue;
                total += m.QuantityKg;
                reused += m.QuantityKg * m.ReuseShare;
            }

            return total > 1e-9 ? reused / total : 0;
        }

        public static double CalculateReclamationPotential(DesignConcept concept)
        {
            if (concept?.Materials == null) return 0;
            double total = concept.Materials.Sum(m => m?.QuantityKg ?? 0);
            if (total <= 1e-9) return 0;
            double reusableMass = concept.Materials.Sum(m => (m?.QuantityKg ?? 0) * concept.FutureReuseProbability);
            return reusableMass / total;
        }

        // ── Top-level summary ──────────────────────────────────────────────────
        /// <summary>Runs the full module-wise LCA for one concept with one allocation method.</summary>
        public static SustainabilityResult CalculateConcept(
            DesignConcept concept,
            AllocationMethod method,
            double numberOfCycles = 2,
            double endOfLifeFactor = 0.03,
            double substitutedEmissionFactor = 0.2,
            double moduleDProcessingCO2 = 0.0)
        {
            if (concept == null)
                return new SustainabilityResult();

            double a1a3 = 0, a4 = 0, a5 = 0, c1c4 = 0;

            foreach (var m in concept.Materials ?? new List<MaterialItem>())
            {
                if (m == null) continue;
                a1a3 += CalculateA1A3(m, method, numberOfCycles);
                a4 += CalculateA4(m);
                a5 += CalculateA5(m);
                c1c4 += CalculateC1C4(m, endOfLifeFactor);
            }

            double connectionCO2 = 0;
            int interlockingWins = 0, interlockingTotal = 0;

            foreach (var c in concept.Connections ?? new List<ConnectionItem>())
            {
                if (c == null) continue;
                if (c.SteelMassKg > 0) connectionCO2 += CalculateSteelConnection(c);
                if (c.ExtraTimberKg > 0)
                {
                    connectionCO2 += CalculateInterlockingConnection(c);
                    interlockingTotal++;
                    if (IsInterlockingBetterThanSteel(c)) interlockingWins++;
                }
            }

            a1a3 += connectionCO2;

            double reusableMass = (concept.Materials ?? new List<MaterialItem>())
                                  .Sum(m => (m?.QuantityKg ?? 0) * (m?.ReuseShare ?? 0));

            double moduleD = CalculateModuleD(
                reusableMass,
                substitutedEmissionFactor,
                concept.SubstitutionFactor,
                concept.FutureReuseProbability,
                moduleDProcessingCO2);

            double totalA1A5 = a1a3 + a4 + a5;
            double totalA1C4 = totalA1A5 + c1c4;

            double area = concept.FunctionalUnitAreaM2 > 1e-9 ? concept.FunctionalUnitAreaM2 : 1.0;
            double years = concept.ReferenceStudyPeriodYears > 1e-9 ? concept.ReferenceStudyPeriodYears : 1.0;

            return new SustainabilityResult
            {
                ConceptName = concept.Name,
                AllocationMethod = method.ToString(),

                A1A3 = a1a3,
                A4 = a4,
                A5 = a5,
                C1C4 = c1c4,
                ModuleD = moduleD,

                TotalA1A5 = totalA1A5,
                TotalA1C4 = totalA1C4,
                OptionalTotalWithD = totalA1C4 + moduleD,

                CO2PerM2 = totalA1C4 / area,
                CO2PerServiceYear = totalA1C4 / years,

                ReuseRatio = CalculateReuseRatio(concept.Materials ?? new List<MaterialItem>()),
                UtilisationRatio = concept.UtilisationRatio,
                ReclamationPotential = CalculateReclamationPotential(concept),

                InterlockingBreakEven = interlockingTotal > 0
                    ? $"{interlockingWins} / {interlockingTotal}"
                    : ""
            };
        }
    }
}
