def MarchingCubes(x,y,z,c,iso,colors):


    PlotFlag = 0
    calc_cols = False
    lindex = 4

    edgeTable, triTable = GetTables()


    if x.ndim != 3 and y.ndim  != 3 and z.ndim  != 3 and c.ndim  != 3:
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

    n = np.array(c.shape) - 1

    cc = np.zeros((n[0],n[1],n[2]))
    cc = cc.astype(np.int32)

    newC = c[0:n[0],0:n[1],0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,1)

    newC = c[1:n[0]+1,0:n[1],0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,2)

    newC = c[1:n[0]+1,1:n[1]+1,0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1)   
    cc = bitset(cc,idx,3)

    newC = c[0:n[0],1:n[1]+1,0:n[2]]              
    idx = newC > iso
    idx = np.multiply(idx, 1) 
    cc = bitset(cc,idx,4)

    newC = c[0:n[0],0:n[1],1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)    
    cc = bitset(cc,idx,5)

    newC = c[1:n[0]+1,0:n[1],1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,6)

    newC = c[1:n[0]+1,1:n[1]+1,1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,7)

    newC = c[0:n[0],1:n[1]+1,1:n[2]+1]              
    idx = newC > iso
    idx = np.multiply(idx, 1)
    cc = bitset(cc,idx,8)


    
    cc2 = cc+1

    cedge = np.zeros(cc2.shape) - 1
    for i in range(0,cc2.shape[0]):
        for j in range(0,cc2.shape[1]):
            for k in range(0,cc2.shape[2]):
                cedge[i,j,k] = edgeTable[cc2[i,j,k]-1]

    cedgeFlatten = cedge.flatten(order='F')
    id =  np.where(cedgeFlatten>0)[0]   
    
    if len(id) == 0:         
        F = []
        V = []
        col = []
        return
    


    xyz_off = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1], [1, 1, 2],  [2, 1, 2], [2, 2, 2], [1, 2, 2]])
    edges = np.array([[1, 2], [2, 3], [3, 4], [4 ,1], [5, 6], [6 ,7], [7, 8], [8, 5], [1 ,5], [2, 6], [3, 7], [4, 8]])

    # xyz_off = xyz_off - 1

    offset = sub2ind(c.shape, xyz_off[:,0], xyz_off[:,1], xyz_off[:,2])
    offset = offset -1
    pp = np.zeros((len(id), lindex, 12))
    
    ccedge = np.column_stack((cedgeFlatten[id], np.array(id)))
    ix_offset=0
        
    xFlatten = x.flatten(order='F')
    yFlatten = y.flatten(order='F')
    zFlatten = z.flatten(order='F')
    cFlatten = c.flatten(order='F')

    for jj in range(0,12):     
        id__ = list(bitget(ccedge[:, 0], jj+1))   
        id__ = [int(item) for item in id__]
        idd = list(np.where(np.array(id__) == 1)[0])
        ccedge_id = ccedge[:,1]
        id_ = np.array(ccedge_id[np.array(id__) == 1]).astype(np.int64)
        ix, iy, iz = ind2sub(cc,list(id_))
        id_c = sub2ind(c.shape, ix+1, iy+1, iz+1) - 1
        id1 = list(id_c + offset[edges[jj, 0]-1])
        id2 = list(id_c + offset[edges[jj, 1]-1])

 

        if calc_cols == True:
            colorsFlatten = colors.flatten(order='F')

            interpolate_val = InterpolateVertices(iso, xFlatten[id1], yFlatten[id1], zFlatten[id1],
                xFlatten[id2], yFlatten[id2], zFlatten[id2], cFlatten[id1], cFlatten[id2], colorsFlatten[id1], colorsFlatten[id2])

            nextp = np.transpose( np.arange(1,id_.shape[0]+1,1)) + ix_offset
            nextp = np.expand_dims(nextp,axis=-1).astype(np.float64)
            pp[idd, :, jj] = np.column_stack((interpolate_val, nextp ))

        else:
            interpolate_val = InterpolateVertices(iso, xFlatten[id1], yFlatten[id1], zFlatten[id1],
                xFlatten[id2], yFlatten[id2], zFlatten[id2], cFlatten[id1], cFlatten[id2])
            
            nextp = np.transpose( np.arange(1,id_.shape[0]+1,1)) + ix_offset
            nextp = np.expand_dims(nextp,axis=-1).astype(np.float64)
            pp[idd, :, jj] = np.column_stack((interpolate_val, nextp ))
            # print(pp[:, :, jj])
            
        # pd.DataFrame(pp[:, :, jj]).to_csv('asd_'+str(jj)+'.csv')
        ix_offset = ix_offset + id_.shape[0]
        # print(np.nanmean(pp))
    # print(np.nanmean(pp))
    # F = []
    cc_flatten = cc.flatten(order='F')
    ab = cc_flatten[id] +1 

    tri = np.zeros((ab.shape[0],triTable.shape[1]))
    for i in range(0,tri.shape[0]):
        for j in range(0,tri.shape[1]):
            tri[i,j] = triTable[ab[i]-1,j]

    # tri = triTable[,:]

    pp_flatten = pp.flatten(order='F')

    for jj in range(0,15,3) :
        id_ = np.where(tri[:, jj]>0)[0]
        
        V = np.array(np.column_stack((id_ + 1, lindex*np.ones((id_.shape[0], 1)),tri[id_,jj:jj+3] )))

        if len(V) > 0:
            p1 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,2]).astype(np.int64) - 1)
            p2 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,3]).astype(np.int64) - 1)
            p3 = list(sub2ind(pp.shape, V[:,0], V[:,1], V[:,4]).astype(np.int64) - 1)

            F2 = np.column_stack((pp_flatten[p1], pp_flatten[p2], pp_flatten[p3]))
            if jj == 0:
                F = F2.copy()
            else:
                F = np.row_stack((F,F2))


    V = []
    col = []
    for jj in range(0,12) :
        idp = list(  np.where(pp[:, lindex-1, jj] > 0)[0]   )
        if np.any(np.array(idp)):
            new_V = list(pp[idp, lindex-1, jj].astype(np.int64))
            # V[new_V, 0:3] = pp[idp, 0:3, jj]
            V.append(pp[idp, 0:3, jj])
            if calc_cols == True:
                new_V = list(pp[idp, lindex-1, jj].astype(np.int64))
                # col[new_V,0] = pp[idp, 3, jj]
                col.append(pp[idp, 3, jj])

    V = np.row_stack(V)
    if len(col) > 0:
        col = np.row_stack(col)

    temp = V.copy()
    temp[np.isnan(V)] = np.inf
    I = np.lexsort((temp[:, 2], temp[:, 1], temp[:, 0])) 
    V = V[I]

    aa = np.diff(V,axis=0)
    bb = np.any(aa>0,axis=1)
    M = list(  np.insert (bb,0,True)    )
    idd = list(np.where(np.array(M) == True)[0])

    V = V[idd,:]
    M = np.multiply(M,1)
    I_rep = np.cumsum(M)
    I2 = np.zeros((I.shape[0],1))
    for i in range(0,I2.shape[0]):
        I2[I[i],0] = I_rep[i]

    F2 = np.zeros((F.shape[0],F.shape[1]))
    for i in range(0,F2.shape[0]):
        F2[i,0] = I2[int(F[i,0])-1]
        F2[i,1] = I2[int(F[i,1])-1]
        F2[i,2] = I2[int(F[i,2])-1]


    return F2,V,col
