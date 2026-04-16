using System;
using System.Collections.Generic;

namespace StructuralCircleNTNU.Classes
{
    public class Project
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Location { get; set; }
        public SupplyBank Supply { get; set; }
        public DemandBank Demand { get; set; }

        public Project() 
        {
            Supply = new SupplyBank();
            Demand = new DemandBank();
        }

        public Project(int id, string name, string location)
        {
            Id = id;
            Name = name;
            Location = location;
            Supply = new SupplyBank();
            Demand = new DemandBank();
        }

        public override string ToString()
        {
            return $"Project [{Id}]: {Name} @ {Location} (Supply: {Supply.Count}, Demand: {Demand.Count})";
        }
    }
}
