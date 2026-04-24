using System;
using System.Collections.Generic;
using System.Linq;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Packing / bin-packing heuristics used by the *Packing matching modes.
    /// 1D cutting stock (beams), 2D shelf packing (plates), 3D bounding-box volume packing
    /// (mixed / generic elements). Deliberately kept as classical deterministic heuristics
    /// (First-Fit-Decreasing / Best-Fit-Decreasing, shelf) so it builds without any external
    /// LP solver — the same family of algorithms that Google OR-Tools uses for pack.BinPacking.
    /// </summary>
    public static class PackingEngine
    {
        const double Tol = 1e-9;

        /// <summary>One packed item placed inside one supply element.</summary>
        public sealed class PackedItem
        {
            public int SupplyIndex { get; set; }
            public int DemandIndex { get; set; }
            /// <summary>Placement in supply-local coordinates (see <see cref="MatchPair.Placement"/>).</summary>
            public Line Placement { get; set; } = Line.Unset;
        }

        /// <summary>Complete packing assignment for one matching run.</summary>
        public sealed class PackingResult
        {
            public List<PackedItem> Items { get; set; } = new List<PackedItem>();
            public HashSet<int> UnpackedDemand { get; set; } = new HashSet<int>();
            public HashSet<int> TouchedSupply { get; set; } = new HashSet<int>();
        }

        // ── 1D cutting stock — First-Fit-Decreasing (FFD) ───────────────────────
        /// <summary>
        /// Pack demand lengths into supply lengths (classic cutting-stock / 1D bin packing).
        /// Feasibility is delegated to <paramref name="feasible"/> so section / Iy / Iz checks happen there.
        /// </summary>
        public static PackingResult Pack1D_FFD(
            double[] supplyLengths,
            double[] demandLengths,
            Func<int, int, bool> feasible)
        {
            int nS = supplyLengths?.Length ?? 0;
            int nD = demandLengths?.Length ?? 0;
            var result = new PackingResult();
            if (nS == 0 || nD == 0)
            {
                for (int i = 0; i < nD; i++) result.UnpackedDemand.Add(i);
                return result;
            }

            double[] remaining = (double[])supplyLengths.Clone();
            double[] offsetTaken = new double[nS];

            var order = Enumerable.Range(0, nD)
                .OrderByDescending(i => demandLengths[i])
                .ToArray();

            foreach (int d in order)
            {
                double dLen = demandLengths[d];
                if (dLen <= Tol) { continue; }

                int chosen = -1;
                for (int s = 0; s < nS; s++)
                {
                    if (!feasible(d, s)) continue;
                    if (remaining[s] + Tol >= dLen) { chosen = s; break; }
                }

                if (chosen < 0) { result.UnpackedDemand.Add(d); continue; }

                double off = offsetTaken[chosen];
                result.Items.Add(new PackedItem
                {
                    SupplyIndex = chosen,
                    DemandIndex = d,
                    Placement = new Line(new Point3d(off, 0, 0), new Point3d(off + dLen, 0, 0))
                });
                result.TouchedSupply.Add(chosen);
                offsetTaken[chosen] += dLen;
                remaining[chosen] -= dLen;
            }

            return result;
        }

        // ── 1D cutting stock — Best-Fit-Decreasing (BFD), score-aware ──────────
        /// <summary>
        /// BFD variant: for each demand pick the supply that leaves the smallest leftover
        /// (tightest fit) among feasible ones with score &lt; ∞. Used by MILP-packing mode.
        /// </summary>
        public static PackingResult Pack1D_BFD(
            double[] supplyLengths,
            double[] demandLengths,
            Func<int, int, bool> feasible,
            Func<int, int, double> score)
        {
            int nS = supplyLengths?.Length ?? 0;
            int nD = demandLengths?.Length ?? 0;
            var result = new PackingResult();
            if (nS == 0 || nD == 0)
            {
                for (int i = 0; i < nD; i++) result.UnpackedDemand.Add(i);
                return result;
            }

            double[] remaining = (double[])supplyLengths.Clone();
            double[] offsetTaken = new double[nS];

            var order = Enumerable.Range(0, nD)
                .OrderByDescending(i => demandLengths[i])
                .ToArray();

            foreach (int d in order)
            {
                double dLen = demandLengths[d];
                if (dLen <= Tol) continue;

                int chosen = -1;
                double bestLeftover = double.MaxValue;
                double bestScore = double.MaxValue;
                for (int s = 0; s < nS; s++)
                {
                    if (!feasible(d, s)) continue;
                    if (remaining[s] + Tol < dLen) continue;
                    double leftover = remaining[s] - dLen;
                    double sc = score(d, s);
                    if (leftover < bestLeftover - Tol ||
                        (Math.Abs(leftover - bestLeftover) <= Tol && sc < bestScore))
                    {
                        bestLeftover = leftover;
                        bestScore = sc;
                        chosen = s;
                    }
                }

                if (chosen < 0) { result.UnpackedDemand.Add(d); continue; }

                double off = offsetTaken[chosen];
                result.Items.Add(new PackedItem
                {
                    SupplyIndex = chosen,
                    DemandIndex = d,
                    Placement = new Line(new Point3d(off, 0, 0), new Point3d(off + dLen, 0, 0))
                });
                result.TouchedSupply.Add(chosen);
                offsetTaken[chosen] += dLen;
                remaining[chosen] -= dLen;
            }

            return result;
        }

        // ── 2D shelf packing ────────────────────────────────────────────────────
        /// <summary>Axis-aligned 2D dimensions for shelf packing.</summary>
        public sealed class Rect2D
        {
            public double W; // along length axis
            public double H; // along width axis (perpendicular in-plane)
        }

        /// <summary>
        /// Classical shelf FFD: open shelves along W, place items left-to-right; when the
        /// current shelf overflows, open a new shelf above. Simple but effective for plates
        /// with matching thickness.
        /// </summary>
        public static PackingResult Pack2D_Shelf(
            Rect2D[] supply,
            Rect2D[] demand,
            Func<int, int, bool> feasible)
        {
            int nS = supply?.Length ?? 0;
            int nD = demand?.Length ?? 0;
            var result = new PackingResult();
            if (nS == 0 || nD == 0)
            {
                for (int i = 0; i < nD; i++) result.UnpackedDemand.Add(i);
                return result;
            }

            var shelves = new List<List<Shelf>>(nS);
            for (int s = 0; s < nS; s++) shelves.Add(new List<Shelf>());

            var order = Enumerable.Range(0, nD)
                .OrderByDescending(i => demand[i].H)
                .ThenByDescending(i => demand[i].W)
                .ToArray();

            foreach (int d in order)
            {
                var item = demand[d];
                bool placed = false;
                for (int s = 0; s < nS && !placed; s++)
                {
                    if (!feasible(d, s)) continue;
                    var sup = supply[s];
                    if (item.W > sup.W + Tol || item.H > sup.H + Tol) continue;

                    foreach (var shelf in shelves[s])
                    {
                        if (item.H <= shelf.Height + Tol &&
                            shelf.UsedW + item.W <= sup.W + Tol)
                        {
                            double x = shelf.UsedW;
                            double y = shelf.Y;
                            result.Items.Add(new PackedItem
                            {
                                SupplyIndex = s,
                                DemandIndex = d,
                                Placement = new Line(new Point3d(x, y, 0), new Point3d(x + item.W, y + item.H, 0))
                            });
                            result.TouchedSupply.Add(s);
                            shelf.UsedW += item.W;
                            placed = true;
                            break;
                        }
                    }
                    if (placed) break;

                    double yTop = shelves[s].Count == 0 ? 0 : shelves[s].Last().Y + shelves[s].Last().Height;
                    if (yTop + item.H <= sup.H + Tol)
                    {
                        var shelf = new Shelf { Y = yTop, Height = item.H, UsedW = item.W };
                        shelves[s].Add(shelf);
                        result.Items.Add(new PackedItem
                        {
                            SupplyIndex = s,
                            DemandIndex = d,
                            Placement = new Line(new Point3d(0, yTop, 0), new Point3d(item.W, yTop + item.H, 0))
                        });
                        result.TouchedSupply.Add(s);
                        placed = true;
                    }
                }

                if (!placed) result.UnpackedDemand.Add(d);
            }

            return result;
        }

        sealed class Shelf
        {
            public double Y;
            public double Height;
            public double UsedW;
        }

        // ── 3D bounding-box packing ─────────────────────────────────────────────
        /// <summary>Axis-aligned 3D dimensions.</summary>
        public sealed class Box3D
        {
            public double DX;
            public double DY;
            public double DZ;
            public double Volume => DX * DY * DZ;
        }

        /// <summary>
        /// 3D BBox packing via greedy extreme-point heuristic: place demand boxes in supply
        /// boxes using a simple corner-point first-fit strategy (not optimal, but a good
        /// sanity-check for mixed element types / brep fallback).
        /// </summary>
        public static PackingResult Pack3D_BBoxFFD(
            Box3D[] supply,
            Box3D[] demand,
            Func<int, int, bool> feasible)
        {
            int nS = supply?.Length ?? 0;
            int nD = demand?.Length ?? 0;
            var result = new PackingResult();
            if (nS == 0 || nD == 0)
            {
                for (int i = 0; i < nD; i++) result.UnpackedDemand.Add(i);
                return result;
            }

            var corners = new List<List<Point3d>>(nS);
            var placed = new List<List<PlacedBox>>(nS);
            for (int s = 0; s < nS; s++)
            {
                corners.Add(new List<Point3d> { Point3d.Origin });
                placed.Add(new List<PlacedBox>());
            }

            var order = Enumerable.Range(0, nD)
                .OrderByDescending(i => demand[i].Volume)
                .ToArray();

            foreach (int d in order)
            {
                var item = demand[d];
                bool done = false;
                for (int s = 0; s < nS && !done; s++)
                {
                    if (!feasible(d, s)) continue;
                    var sup = supply[s];
                    if (item.DX > sup.DX + Tol || item.DY > sup.DY + Tol || item.DZ > sup.DZ + Tol) continue;

                    foreach (var corner in corners[s].OrderBy(p => p.X).ThenBy(p => p.Y).ThenBy(p => p.Z).ToList())
                    {
                        var min = corner;
                        var max = new Point3d(min.X + item.DX, min.Y + item.DY, min.Z + item.DZ);
                        if (max.X > sup.DX + Tol || max.Y > sup.DY + Tol || max.Z > sup.DZ + Tol) continue;
                        if (Overlaps(placed[s], min, max)) continue;

                        placed[s].Add(new PlacedBox { Min = min, Max = max });
                        corners[s].Remove(corner);
                        corners[s].Add(new Point3d(max.X, min.Y, min.Z));
                        corners[s].Add(new Point3d(min.X, max.Y, min.Z));
                        corners[s].Add(new Point3d(min.X, min.Y, max.Z));

                        result.Items.Add(new PackedItem
                        {
                            SupplyIndex = s,
                            DemandIndex = d,
                            Placement = new Line(min, max)
                        });
                        result.TouchedSupply.Add(s);
                        done = true;
                        break;
                    }
                }

                if (!done) result.UnpackedDemand.Add(d);
            }

            return result;
        }

        sealed class PlacedBox
        {
            public Point3d Min;
            public Point3d Max;
        }

        static bool Overlaps(List<PlacedBox> boxes, Point3d min, Point3d max)
        {
            foreach (var b in boxes)
            {
                if (min.X >= b.Max.X - Tol || max.X <= b.Min.X + Tol) continue;
                if (min.Y >= b.Max.Y - Tol || max.Y <= b.Min.Y + Tol) continue;
                if (min.Z >= b.Max.Z - Tol || max.Z <= b.Min.Z + Tol) continue;
                return true;
            }
            return false;
        }

        // ── Helpers for building dimension arrays ──────────────────────────────
        public static double GetLength(Element e)
        {
            if (e is Beam b) return b.Length;
            if (e is Plate p) return p.Length;
            return e != null && e.AxisLine.IsValid ? e.AxisLine.Length : 0;
        }

        public static Rect2D GetPlateRect(Element e)
        {
            if (e is Plate p)
            {
                double w = p.Section?.Width ?? 0;
                double l = p.Length;
                return new Rect2D { W = Math.Max(l, 1e-6), H = Math.Max(w, 1e-6) };
            }
            return new Rect2D { W = 1e-6, H = 1e-6 };
        }

        public static Box3D GetBbox(Element e)
        {
            if (e is Beam b && b.Section is BeamSection bs)
                return new Box3D { DX = b.Length, DY = bs.Width, DZ = bs.Height };
            if (e is Plate p && p.Section is PlateSection ps)
                return new Box3D { DX = p.Length, DY = ps.Width, DZ = ps.Thickness };
            if (e.GeometryBrep != null && e.GeometryBrep.IsValid)
            {
                var bb = e.GeometryBrep.GetBoundingBox(true);
                return new Box3D
                {
                    DX = bb.Max.X - bb.Min.X,
                    DY = bb.Max.Y - bb.Min.Y,
                    DZ = bb.Max.Z - bb.Min.Z
                };
            }
            return new Box3D { DX = 0, DY = 0, DZ = 0 };
        }
    }
}
