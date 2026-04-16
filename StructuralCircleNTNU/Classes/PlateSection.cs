using System;

namespace StructuralCircleNTNU.Classes
{
    public class PlateSection : Section
    {
        public double Thickness { get; set; }

        /// <summary>Panel / slab width in metres (the in-plane span perpendicular to the length axis).</summary>
        public double Width { get; set; }

        public PlateSection() { }

        public PlateSection(int id, string name, double thickness)
            : base(id, name)
        {
            Thickness = thickness;
        }

        public PlateSection(int id, string name, double thickness, double width)
            : base(id, name)
        {
            Thickness = thickness;
            Width = width;
        }

        public override string ToString()
        {
            return Width > 0
                ? $"PlateSection [{Id}]: {Name} (T={Thickness:F3}, W={Width:F3})"
                : $"PlateSection [{Id}]: {Name} (T={Thickness:F3})";
        }
    }
}
