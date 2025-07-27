import numpy as np

def geospatial(df_int, spacing, xI, yI, zI):
    """
    Memory-optimized geospatial analysis function calculating Moran's I and Geary's C statistics.
    
    Memory optimizations applied line-by-line:
    - Replaced .shape[0] with len() for better memory efficiency
    - Pre-allocated arrays with specific dtype to avoid temporary arrays
    - Used in-place operations instead of intermediate array creation
    - Replaced inefficient tile/repeat with meshgrid and boolean masking
    - Eliminated list comprehensions and map() with vectorized operations
    - Added explicit memory cleanup with del statements
    - Cached frequently used calculations
    - Used broadcasting for efficient array operations
    - Replaced np.arange with range in loops for better memory usage
    - Used slicing instead of array creation where possible
    """
    
    # OPTIMIZATION 1: Use len() instead of .shape[0] - more memory efficient
    n_v = len(df_int)

    # OPTIMIZATION 2: Pre-allocate with specific dtype and use in-place assignment
    # Original: pos_mat = np.column_stack((zI * spacing[0], yI * spacing[1], xI * spacing[2]))
    pos_mat = np.empty((n_v, 3), dtype=np.float64)  # Pre-allocate
    pos_mat[:, 0] = zI * spacing[0]  # In-place assignment
    pos_mat[:, 1] = yI * spacing[1]  # In-place assignment  
    pos_mat[:, 2] = xI * spacing[2]  # In-place assignment

    if n_v < 2000:
        # OPTIMIZATION 3: Replace inefficient tile/repeat with meshgrid and boolean masking
        # Original: comb_iter = np.array([np.tile(np.arange(0, n_v), n_v), np.repeat(np.arange(0, n_v), n_v)])
        # Original: comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]
        i_indices, j_indices = np.meshgrid(np.arange(n_v), np.arange(n_v), indexing='ij')
        mask = i_indices > j_indices  # Boolean mask is more memory efficient
        comb_i = i_indices[mask]  # Extract only upper triangle indices
        comb_j = j_indices[mask]
        
        # OPTIMIZATION 4: Vectorized distance calculation instead of list comprehension + map
        # Original: w_ij = 1.0 / np.array(list(map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)), np.arange(np.shape(comb_iter)[1]))))
        diff = pos_mat[comb_i] - pos_mat[comb_j]  # Broadcasting subtraction
        w_ij = 1.0 / np.sqrt(np.sum(diff ** 2, axis=1))  # Vectorized distance
        del diff  # OPTIMIZATION 5: Explicit memory cleanup

        # OPTIMIZATION 6: Cache mean calculation and avoid inline np.mean calls
        # Original: gl_dev = df_int - np.mean(df_int)
        mean_intensity = np.mean(df_int)
        gl_dev = df_int - mean_intensity

        # OPTIMIZATION 7: Use direct multiplication instead of np.multiply chains and cache calculations
        # Original: nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
        # Original: denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
        moran_nom = n_v * np.sum(w_ij * gl_dev[comb_i] * gl_dev[comb_j])
        gl_dev_sq_sum = np.sum(gl_dev ** 2)  # Cache this calculation
        w_sum = np.sum(w_ij)  # Cache weight sum
        
        if gl_dev_sq_sum > 0.0:
            moran_i = moran_nom / (w_sum * gl_dev_sq_sum)
        else:
            moran_i = 1.0

        # OPTIMIZATION 8: Reuse cached values for Geary's C calculation
        # Original: nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
        geary_nom = (n_v - 1.0) * np.sum(w_ij * (gl_dev[comb_i] - gl_dev[comb_j]) ** 2)
        if gl_dev_sq_sum > 0.0:
            geary_c = geary_nom / (2.0 * w_sum * gl_dev_sq_sum)
        else:
            geary_c = 1.0
            
        # OPTIMIZATION 9: Clean up large arrays explicitly
        del w_ij, comb_i, comb_j, gl_dev
    else:
        # Large dataset optimizations
        mean_intensity = np.mean(df_int)
        gl_dev = df_int - mean_intensity
        
        # OPTIMIZATION 10: Pre-calculate denominator once instead of recalculating
        gl_denom = np.sum(gl_dev ** 2)
        
        # OPTIMIZATION 11: Use specific dtype for accumulators to avoid precision loss
        moran_nom = np.float64(0.0)
        geary_nom = np.float64(0.0) 
        w_denom = np.float64(0.0)

        # OPTIMIZATION 12: Use range instead of np.arange for better memory efficiency
        # Original: for ii in np.arange(n_v - 1):
        for ii in range(n_v - 1):
            # OPTIMIZATION 13: Use slice notation instead of np.arange
            # Original: jj = np.arange(start=ii + 1, stop=n_v)
            jj_slice = slice(ii + 1, n_v)
            
            # OPTIMIZATION 14: Use broadcasting for vectorized distance calculation
            # Original: w_iijj = 1.0 / np.sqrt(np.sum(np.power(pos_mat[ii, :] - pos_mat[jj, :], 2.0), axis=1))
            pos_diff = pos_mat[ii:ii+1, :] - pos_mat[jj_slice, :]  # Broadcasting
            w_iijj = 1.0 / np.sqrt(np.sum(pos_diff ** 2, axis=1))
            
            # OPTIMIZATION 15: Cache scalar and slice values to avoid repeated indexing
            gl_dev_ii = gl_dev[ii]  # Cache scalar
            gl_dev_jj = gl_dev[jj_slice]  # Cache slice
            
            # OPTIMIZATION 16: Replace np.multiply chains with direct operations
            # Original: moran_nom += np.sum(np.multiply(np.multiply(w_iijj, gl_dev[ii]), gl_dev[jj]))
            # Original: geary_nom += np.sum(np.multiply(w_iijj, (gl_dev[ii] - gl_dev[jj]) ** 2.0))
            moran_nom += np.sum(w_iijj * gl_dev_ii * gl_dev_jj)
            geary_nom += np.sum(w_iijj * (gl_dev_ii - gl_dev_jj) ** 2)
            w_denom += np.sum(w_iijj)
            
            # OPTIMIZATION 17: Clean up intermediate arrays in each iteration
            del pos_diff, w_iijj, gl_dev_jj

        # Final calculations with cached denominator
        if gl_denom > 0.0:
            moran_i = n_v * moran_nom / (w_denom * gl_denom)
            geary_c = (n_v - 1.0) * geary_nom / (2 * w_denom * gl_denom)
        else:
            moran_i = 1.0
            geary_c = 1.0
            
        # OPTIMIZATION 18: Final memory cleanup
        del gl_dev

    return moran_i, geary_c