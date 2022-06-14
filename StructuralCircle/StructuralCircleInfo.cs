using Grasshopper;
using Grasshopper.Kernel;
using System;
using System.Drawing;

namespace StructuralCircle
{
    public class StructuralCircleInfo : GH_AssemblyInfo
    {
        public override string Name => "StructuralCircle";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => null;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "";

        public override Guid Id => new Guid("8D97DACC-0512-4E35-8BCC-CC24CF60C150");

        //Return a string identifying you or your company.
        public override string AuthorName => "";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "";
    }
}