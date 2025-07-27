import tempfile
import numpy as np
import os


def chunked_copy(array, chunk_size=None):
    """
    Memory-efficient chunked copying of numpy arrays.
    
    Args:
        array: Input numpy array
        chunk_size: Size of chunks for processing. If None, auto-calculated based on array size and dtype
    
    Returns:
        Copy of the array (either in-memory or memory-mapped)
    """
    # Auto-calculate optimal chunk size if not provided
    if chunk_size is None:
        # Calculate chunk size based on array dtype and available memory considerations
        # Aim for ~100MB chunks by default, but adjust based on array size
        bytes_per_element = array.dtype.itemsize
        target_chunk_bytes = 100 * 1024 * 1024  # 100MB
        chunk_size = max(1000, min(array.size, target_chunk_bytes // bytes_per_element))
    
    # For small arrays, use regular copy
    if array.size <= chunk_size:
        return array.copy()
    
    # For large arrays, use memory mapping with optimizations
    temp_file = None
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()  # Close file handle to avoid issues on Windows
        
        # Create memory-mapped output array
        out = np.memmap(temp_file.name, dtype=array.dtype, mode='w+', shape=array.shape)
        
        # Use views instead of flat indexing for better performance
        if array.ndim == 1:
            # For 1D arrays, direct slicing is most efficient
            for i in range(0, array.size, chunk_size):
                end_idx = min(i + chunk_size, array.size)
                out[i:end_idx] = array[i:end_idx]
        else:
            # For multi-dimensional arrays, use raveled views
            array_flat = array.ravel()  # Creates a view, not a copy
            out_flat = out.ravel()      # Creates a view, not a copy
            
            for i in range(0, array.size, chunk_size):
                end_idx = min(i + chunk_size, array.size)
                # Use views for memory efficiency
                out_flat[i:end_idx] = array_flat[i:end_idx]
        
        # Ensure data is written to disk
        out.flush()
        
        return out
        
    except Exception as e:
        # Clean up on error
        if temp_file is not None and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass  # Ignore cleanup errors
        raise e