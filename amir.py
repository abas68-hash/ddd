def MarchingCubes(x,y,z,c,iso,colors):
    import numpy as np
    
    PlotFlag = 0
    calc_cols = False
    lindex = 4

    edgeTable, triTable = GetTables()

    # Input validation - use 'or' instead of 'and' for correct logic
    if x.ndim != 3 or y.ndim != 3 or z.ndim != 3 or c.ndim != 3:
        raise ValueError('x, y, z, c must be matrices of dim 3')
    
    if x.shape != y.shape or y.shape != z.shape or z.shape != c.shape:
        raise ValueError('x, y, z, c must be the same size')
    
    if np.any(x.shape < (2, 2, 2)):
        raise ValueError('grid size must be at least 2x2x2')
    
    if colors is not None:
        if colors.shape != c.shape:
            raise ValueError('color must be matrix of same size as c')
        calc_cols = True
        lindex = 5

    n = np.array(c.shape, dtype=np.int32) - 1

    # Pre-allocate with appropriate data type
    cc = np.zeros((n[0], n[1], n[2]), dtype=np.uint8)

    # Vectorized cube configuration computation - avoid temporary arrays
    # Process all 8 cube vertices at once using advanced indexing
    cube_slices = [
        (slice(0, n[0]), slice(0, n[1]), slice(0, n[2])),       # vertex 0
        (slice(1, n[0]+1), slice(0, n[1]), slice(0, n[2])),     # vertex 1
        (slice(1, n[0]+1), slice(1, n[1]+1), slice(0, n[2])),   # vertex 2
        (slice(0, n[0]), slice(1, n[1]+1), slice(0, n[2])),     # vertex 3
        (slice(0, n[0]), slice(0, n[1]), slice(1, n[2]+1)),     # vertex 4
        (slice(1, n[0]+1), slice(0, n[1]), slice(1, n[2]+1)),   # vertex 5
        (slice(1, n[0]+1), slice(1, n[1]+1), slice(1, n[2]+1)), # vertex 6
        (slice(0, n[0]), slice(1, n[1]+1), slice(1, n[2]+1))    # vertex 7
    ]
    
    # Vectorized bitset operations
    for i, slc in enumerate(cube_slices):
        mask = c[slc] > iso
        cc = bitset(cc, mask.astype(np.uint8), i+1)

    # Vectorized edge table lookup
    cc2 = cc + 1
    cedge = edgeTable[cc2.ravel() - 1].reshape(cc2.shape)
    
    # Find active cubes more efficiently
    active_mask = cedge > 0
    if not np.any(active_mask):
        return [], [], []
    
    # Get flat indices of active cubes
    active_indices = np.flatnonzero(cedge.ravel(order='F') > 0)
    cedge_values = cedge.ravel(order='F')[active_indices]
    
    # Pre-compute constants
    xyz_off = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1], 
                        [1, 1, 2], [2, 1, 2], [2, 2, 2], [1, 2, 2]], dtype=np.int32)
    edges = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7], 
                      [7, 8], [8, 5], [1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.int32)
    
    offset = sub2ind(c.shape, xyz_off[:,0], xyz_off[:,1], xyz_off[:,2]) - 1
    
    # Pre-allocate with smaller data type
    num_active = len(active_indices)
    pp = np.zeros((num_active, lindex, 12), dtype=np.float32)
    
    ccedge = np.column_stack((cedge_values, active_indices))
    ix_offset = 0
    
    # Flatten arrays only once, outside the loop
    x_flat = x.ravel(order='F')
    y_flat = y.ravel(order='F') 
    z_flat = z.ravel(order='F')
    c_flat = c.ravel(order='F')
    colors_flat = colors.ravel(order='F') if calc_cols else None

    for jj in range(12):     
        # Vectorized bit operations
        edge_mask = bitget(ccedge[:, 0], jj+1).astype(bool)
        if not np.any(edge_mask):
            continue
            
        active_edge_indices = np.where(edge_mask)[0]
        cube_indices = ccedge[edge_mask, 1].astype(np.int64)
        
        ix, iy, iz = ind2sub(cc, cube_indices)
        id_c = sub2ind(c.shape, ix+1, iy+1, iz+1) - 1
        
        # Vectorized offset computation
        edge_info = edges[jj] - 1  # Convert to 0-based
        id1 = id_c + offset[edge_info[0]]
        id2 = id_c + offset[edge_info[1]]

        if calc_cols:
            interpolate_val = InterpolateVertices(iso, x_flat[id1], y_flat[id1], z_flat[id1],
                x_flat[id2], y_flat[id2], z_flat[id2], c_flat[id1], c_flat[id2], 
                colors_flat[id1], colors_flat[id2])
        else:
            interpolate_val = InterpolateVertices(iso, x_flat[id1], y_flat[id1], z_flat[id1],
                x_flat[id2], y_flat[id2], z_flat[id2], c_flat[id1], c_flat[id2])
        
        # Efficient index generation
        nextp = (np.arange(len(cube_indices), dtype=np.float32) + 1 + ix_offset).reshape(-1, 1)
        pp[active_edge_indices, :, jj] = np.column_stack((interpolate_val, nextp))
        
        ix_offset += len(cube_indices)

    # Vectorized triangle table lookup
    cc_flat = cc.ravel(order='F')
    ab = cc_flat[active_indices] + 1
    tri = triTable[ab - 1]

    # Pre-allocate faces list more efficiently
    faces_list = []
    pp_flat = pp.ravel(order='F')

    for jj in range(0, 15, 3):
        tri_mask = tri[:, jj] > 0
        if not np.any(tri_mask):
            continue
            
        valid_indices = np.where(tri_mask)[0]
        tri_data = tri[tri_mask, jj:jj+3]
        
        V_temp = np.column_stack((valid_indices + 1, 
                                 np.full(len(valid_indices), lindex), 
                                 tri_data))
        
        # Vectorized sub2ind operations
        p1 = sub2ind(pp.shape, V_temp[:,0], V_temp[:,1], V_temp[:,2]) - 1
        p2 = sub2ind(pp.shape, V_temp[:,0], V_temp[:,1], V_temp[:,3]) - 1
        p3 = sub2ind(pp.shape, V_temp[:,0], V_temp[:,1], V_temp[:,4]) - 1
        
        F2 = np.column_stack((pp_flat[p1], pp_flat[p2], pp_flat[p3]))
        faces_list.append(F2)

    # Concatenate faces more efficiently
    F = np.vstack(faces_list) if faces_list else np.array([])

    # Collect vertices more efficiently
    vertex_list = []
    color_list = []
    
    for jj in range(12):
        vertex_mask = pp[:, lindex-1, jj] > 0
        if np.any(vertex_mask):
            vertex_list.append(pp[vertex_mask, 0:3, jj])
            if calc_cols:
                color_list.append(pp[vertex_mask, 3, jj])

    if not vertex_list:
        return np.array([]), np.array([]), np.array([])
        
    V = np.vstack(vertex_list)
    col = np.vstack(color_list) if color_list else np.array([])

    # More efficient unique vertex finding
    # Use lexsort with NaN handling
    V_temp = V.copy()
    V_temp[np.isnan(V)] = np.inf
    sort_indices = np.lexsort((V_temp[:, 2], V_temp[:, 1], V_temp[:, 0]))
    V_sorted = V[sort_indices]

    # Find unique vertices using vectorized operations
    diff_mask = np.concatenate(([True], np.any(np.diff(V_sorted, axis=0) != 0, axis=1)))
    unique_indices = np.where(diff_mask)[0]
    V_unique = V_sorted[unique_indices]
    
    # Create mapping from old to new indices
    mapping = np.zeros(len(sort_indices), dtype=np.int32)
    mapping[sort_indices] = np.cumsum(diff_mask) - 1
    
    # Update face indices efficiently
    if len(F) > 0:
        F_new = mapping[F.astype(np.int32) - 1]
    else:
        F_new = np.array([])

    return F_new, V_unique, col
