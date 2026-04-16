using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace StructuralCircleNTNU
{
    public class StructuralCircleNTNUInfo : GH_AssemblyInfo
    {
        public override string Name => "StructuralCircleNTNU";

        public override Bitmap Icon => IconLoader.GetIcon();

        public override string Description => "Sustainable design from used structural elements. " +
            "Matching algorithms for reclaimed building components in Grasshopper.";

        public override Guid Id => new Guid("6b45ca7d-c2f2-4d38-9a86-179ee05da3d1");

        public override string AuthorName => "NTNU Structural Circle";

        public override string AuthorContact => "https://github.com/marcinluczkowski/structuralCircle";

        public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
    }
}
