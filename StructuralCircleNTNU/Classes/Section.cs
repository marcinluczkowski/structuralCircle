using System;

namespace StructuralCircleNTNU.Classes
{
    public abstract class Section
    {
        public int Id { get; set; }
        public string Name { get; set; }

        protected Section() { }

        protected Section(int id, string name)
        {
            Id = id;
            Name = name;
        }

        public override string ToString()
        {
            return $"Section [{Id}]: {Name}";
        }
    }
}
