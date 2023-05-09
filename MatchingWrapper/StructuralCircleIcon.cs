using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatchingWrapper
{
    public class StructuralCircleIcon : Grasshopper.Kernel.GH_AssemblyPriority
    {
        public override Grasshopper.Kernel.GH_LoadingInstruction PriorityLoad()
        {
            Grasshopper.Instances.ComponentServer.AddCategoryIcon("Python", Properties.Resources.matching_logo);
            Grasshopper.Instances.ComponentServer.AddCategorySymbolName("Python", 'P');
            return Grasshopper.Kernel.GH_LoadingInstruction.Proceed;
        }
    }
}
