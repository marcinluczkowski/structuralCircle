1. Demand sorted descending
2. Supply sorted ascending
3. Each list should contain:
   {0} Length
   {1} Area
   {2} Mesh
   {3} Moment of inertia
   {4} Line
   {5} Height

1. Set 'Plural_assign':
   - False - only one-to-one mapping
   - True - multiple 'demand' beams can use the same 'supply' beam.
2. for each 'demand' element the algorithm goes through the list of available 'supply' elements until it finds an element of length and area higher or exactly as required.
3. If 'plural_assign' is set to True, it will shorten the mapped beam and sort the 'supply' list after that update. If False, the element is removed from the list after mapping.