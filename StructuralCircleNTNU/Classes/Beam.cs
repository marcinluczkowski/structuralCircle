using System;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    public class Beam : Element
    {
        public override string ElementType => "Beam";

        public new BeamSection Section
        {
            get => base.Section as BeamSection;
            set => base.Section = value;
        }

        public double Length
        {
            get => AxisLine.Length;
        }

        public Beam() { }

        public Beam(int id, string name, string location, Material material, BeamSection section, Line axisLine)
            : base(id, name, location, material, section)
        {
            AxisLine = axisLine;
        }

        public override string ToString()
        {
            string sec = Section != null ? $", {Section}" : "";
            string mat = Material != null ? $", {Material}" : "";
            return $"Beam [{Id}]: {Name} (L={Length:F3}{sec}{mat})";
        }
    }
}
