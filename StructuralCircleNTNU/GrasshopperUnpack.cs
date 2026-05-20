using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU
{
    /// <summary>
    /// Unwraps values from Grasshopper generic parameters (<see cref="GH_ObjectWrapper"/>, <see cref="IGH_Goo"/>)
    /// into plugin domain types.  All methods return null when the value cannot be cast.
    /// </summary>
    internal static class GrasshopperUnpack
    {
        public static Material     AsMaterial    (object d) => Unwrap<Material>(d);
        public static Element      AsElement     (object d) => Unwrap<Element>(d);
        public static Beam         AsBeam        (object d) => Unwrap<Beam>(d);
        public static Plate        AsPlate       (object d) => Unwrap<Plate>(d);
        public static BeamSection  AsBeamSection (object d) => Unwrap<BeamSection>(d);
        public static PlateSection AsPlateSection(object d) => Unwrap<PlateSection>(d);
        public static MatchResult  AsMatchResult (object d) => Unwrap<MatchResult>(d);
        public static MatchPair    AsMatchPair   (object d) => Unwrap<MatchPair>(d);
        public static SupplyBank   AsSupplyBank  (object d) => Unwrap<SupplyBank>(d);
        public static DemandBank   AsDemandBank  (object d) => Unwrap<DemandBank>(d);

        static T Unwrap<T>(object data) where T : class
        {
            if (data == null) return null;
            if (data is T direct) return direct;
            if (data is GH_ObjectWrapper wrap && wrap.Value is T wv) return wv;
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return Unwrap<T>(v);
                }
                catch { }
            }
            return null;
        }
    }
}
