using System;

namespace StructuralCircleNTNU.Classes
{
    public class Material
    {
        public int Id { get; set; }
        public string Name { get; set; }

        public Material() { }

        public Material(int id, string name)
        {
            Id = id;
            Name = name;
        }

        public override string ToString()
        {
            return $"Material [{Id}]: {Name}";
        }
    }
}
