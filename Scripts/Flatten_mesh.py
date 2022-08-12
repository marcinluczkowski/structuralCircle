import rhinoscriptsyntax as rs
 
def FlattenMesh():
    #    arrObjects = rs.GetObjects("Select Objects", rs.filter.mesh, True, True)
    surf_id = rs.GetSurfaceObject(message='Select surface', preselect=False, select=False)
    d = rs.SurfaceClosestPoint(surf_id[0], rs.CreatePoint(0,0,0))
    
    object_id="d8ce05cc-01ad-425f-b232-c93055332bd4"
    vert = rs.GetMeshVertices(object_id, message='', min_count=1, max_count=0)
    
    for v in vert:
        plane = rs.PlaneFromNormal(rs.CreatePoint(0,0,0),rs.CreateVector(0,0,1))
        surf.TryGetPlane(plane)
        pt = rs.CreatePoint(10,20,30)
        dist = rs.DistanceToPlane(plane, pt)
        
        pass
    
    
    
    pass
    
    
    
    
    
if( __name__ == "__main__" ):
    FlattenMesh()