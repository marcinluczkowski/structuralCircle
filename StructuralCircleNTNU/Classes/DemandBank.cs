using System;
using System.Collections.Generic;
using System.Linq;

namespace StructuralCircleNTNU.Classes
{
    public class DemandBank
    {
        public List<Element> Elements { get; set; }

        public DemandBank()
        {
            Elements = new List<Element>();
        }

        public DemandBank(List<Element> elements)
        {
            Elements = elements ?? new List<Element>();
        }

        public void Add(Element element)
        {
            Elements.Add(element);
        }

        public int Count => Elements.Count;

        public List<Beam> Beams => Elements.OfType<Beam>().ToList();
        public List<Plate> Plates => Elements.OfType<Plate>().ToList();

        public override string ToString()
        {
            return $"DemandBank ({Count} elements: {Beams.Count} beams, {Plates.Count} plates)";
        }
    }
}
