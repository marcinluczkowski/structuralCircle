using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using StructuralCircleNTNU.Classes;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU
{
    /// <summary>
    /// Unwraps values from Grasshopper generic parameters (<see cref="GH_ObjectWrapper"/>, <see cref="IGH_Goo"/>)
    /// into plugin domain types.
    /// </summary>
    internal static class GrasshopperUnpack
    {
        public static Material AsMaterial(object data)
        {
            if (data == null) return null;
            if (data is Material mat) return mat;
            if (data is GH_ObjectWrapper wrap && wrap.Value is Material wm)
                return wm;
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsMaterial(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }
            return null;
        }

        public static SupplyBank AsSupplyBank(object data)
        {
            if (data == null) return null;
            if (data is SupplyBank sb) return sb;
            if (data is GH_ObjectWrapper wrap)
                return AsSupplyBank(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsSupplyBank(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static DemandBank AsDemandBank(object data)
        {
            if (data == null) return null;
            if (data is DemandBank db) return db;
            if (data is GH_ObjectWrapper wrap)
                return AsDemandBank(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsDemandBank(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static Classes.MatchResult AsMatchResult(object data)
        {
            if (data == null) return null;
            if (data is Classes.MatchResult mr) return mr;
            if (data is GH_ObjectWrapper wrap)
                return AsMatchResult(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsMatchResult(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static MaterialItem AsMaterialItem(object data)
        {
            if (data == null) return null;
            if (data is MaterialItem mi) return mi;
            if (data is GH_ObjectWrapper wrap)
                return AsMaterialItem(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsMaterialItem(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static ConnectionItem AsConnectionItem(object data)
        {
            if (data == null) return null;
            if (data is ConnectionItem ci) return ci;
            if (data is GH_ObjectWrapper wrap)
                return AsConnectionItem(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsConnectionItem(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static DesignConcept AsDesignConcept(object data)
        {
            if (data == null) return null;
            if (data is DesignConcept dc) return dc;
            if (data is GH_ObjectWrapper wrap)
                return AsDesignConcept(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsDesignConcept(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }

        public static SustainabilityResult AsSustainabilityResult(object data)
        {
            if (data == null) return null;
            if (data is SustainabilityResult sr) return sr;
            if (data is GH_ObjectWrapper wrap)
                return AsSustainabilityResult(wrap.Value);
            if (data is IGH_Goo goo)
            {
                try
                {
                    object v = goo.ScriptVariable();
                    if (v != null && !ReferenceEquals(v, data))
                        return AsSustainabilityResult(v);
                }
                catch { /* ScriptVariable not supported for this goo type */ }
            }

            return null;
        }
    }
}
