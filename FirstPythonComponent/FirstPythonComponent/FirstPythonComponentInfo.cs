using Grasshopper;
using Grasshopper.Kernel;
using System;
using System.Drawing;

namespace FirstPythonComponent
{
    public class FirstPythonComponentInfo : GH_AssemblyInfo
    {
        public override string Name => "FirstPythonComponent";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => null;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "";

        public override Guid Id => new Guid("6E9405C5-FA37-4EB5-B056-6E501B60B53A");

        //Return a string identifying you or your company.
        public override string AuthorName => "";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "";
    }
}