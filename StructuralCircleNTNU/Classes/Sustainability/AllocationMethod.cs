namespace StructuralCircleNTNU.Classes.Sustainability
{
    /// <summary>
    /// LCA allocation method for the reused share of a material (A1–A3 burden split).
    /// 0 = CutOff (reused carries nothing)
    /// 1 = PEF50_50 (half of reused share carries burden)
    /// 2 = NCycle (burden divided across N cycles)
    /// </summary>
    public enum AllocationMethod
    {
        CutOff = 0,
        PEF50_50 = 1,
        NCycle = 2
    }
}
