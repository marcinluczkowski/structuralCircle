using System;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using StructuralCircleNTNU.Classes;

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
    }
}
