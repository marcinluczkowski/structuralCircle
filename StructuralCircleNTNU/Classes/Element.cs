using System;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    public abstract class Element
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Location { get; set; }

        public Brep GeometryBrep { get; set; }
        public Mesh GeometryMesh { get; set; }
        public Line AxisLine { get; set; }
        public Surface AxisSurface { get; set; }

        public Material Material { get; set; }
        public Section Section { get; set; }

        protected Element() { }

        protected Element(int id, string name, string location, Material material, Section section)
        {
            Id = id;
            Name = name;
            Location = location;
            Material = material;
            Section = section;
        }

        public abstract string ElementType { get; }

        public override string ToString()
        {
            return $"{ElementType} [{Id}]: {Name} @ {Location}";
        }
    }
}
