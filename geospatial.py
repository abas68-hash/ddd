def geospatial(df_int, spacing, xI, yI, zI):
    # Define constants
    n_v = df_int.shape[0]

    pos_mat = np.column_stack((zI * spacing[0], yI * spacing[1], xI * spacing[2]))

    if n_v < 2000:
        # Determine all interactions between voxels
        comb_iter = np.array([np.tile(np.arange(0, n_v), n_v), np.repeat(np.arange(0, n_v), n_v)])
        comb_iter = comb_iter[:, comb_iter[0, :] > comb_iter[1, :]]

        # Determine weighting for all interactions (inverse weighting with distance)
        w_ij = 1.0 / np.array(list(
            map(lambda i: np.sqrt(np.sum((pos_mat[comb_iter[0, i], :] - pos_mat[comb_iter[1, i], :]) ** 2.0)),
                np.arange(np.shape(comb_iter)[1]))))

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int - np.mean(df_int)

        # Moran's I
        nom = n_v * np.sum(np.multiply(np.multiply(w_ij, gl_dev[comb_iter[0, :]]), gl_dev[comb_iter[1, :]]))
        denom = np.sum(w_ij) * np.sum(gl_dev ** 2.0)
        if denom > 0.0:
            moran_i = nom / denom
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume, which indicates ideal spatial correlation.
            moran_i = 1.0

        # Geary's C
        nom = (n_v - 1.0) * np.sum(np.multiply(w_ij, (gl_dev[comb_iter[0, :]] - gl_dev[comb_iter[1, :]]) ** 2.0))
        if denom > 0.0:
            geary_c = nom / (2.0 * denom)
        else:
            # If the denominator is 0.0, this basically means only one intensity is present in the volume.
            geary_c = 1.0
    else:
        # In practice, this code variant is only used if the ROI is too large to perform all distance calculations in one go.

        # Create array of mean-corrected grey level intensities
        gl_dev = df_int - np.mean(df_int)

        moran_nom = 0.0
        geary_nom = 0.0
        w_denom = 0.0

        # Iterate over voxels
        for ii in np.arange(n_v - 1):
            # Get all jj > ii voxels
            jj = np.arange(start=ii + 1, stop=n_v)

            # Get distance weights
            w_iijj = 1.0 / np.sqrt(np.sum(np.power(pos_mat[ii, :] - pos_mat[jj, :], 2.0), axis=1))

            moran_nom += np.sum(np.multiply(np.multiply(w_iijj, gl_dev[ii]), gl_dev[jj]))
            geary_nom += np.sum(np.multiply(w_iijj, (gl_dev[ii] - gl_dev[jj]) ** 2.0))
            w_denom += np.sum(w_iijj)

        gl_denom = np.sum(gl_dev ** 2.0)

        # Moran's I index
        if gl_denom > 0.0:
            moran_i = n_v * moran_nom / (w_denom * gl_denom)
        else:
            moran_i = 1.0

        # Geary's C measure
        if gl_denom > 0.0:
            geary_c = (n_v - 1.0) * geary_nom / (2 * w_denom * gl_denom)
        else:
            geary_c = 1.0

    return moran_i, geary_c
