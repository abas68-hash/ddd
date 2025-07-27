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


def chunked_copy_with_progress(array, chunk_size=None, progress_callback=None):
    """
    Memory-efficient chunked copying with optional progress reporting.
    
    Args:
        array: Input numpy array
        chunk_size: Size of chunks for processing
        progress_callback: Optional function to call with progress (0.0 to 1.0)
    
    Returns:
        Copy of the array
    """
    if chunk_size is None:
        bytes_per_element = array.dtype.itemsize
        target_chunk_bytes = 100 * 1024 * 1024  # 100MB
        chunk_size = max(1000, min(array.size, target_chunk_bytes // bytes_per_element))
    
    if array.size <= chunk_size:
        if progress_callback:
            progress_callback(1.0)
        return array.copy()
    
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        out = np.memmap(temp_file.name, dtype=array.dtype, mode='w+', shape=array.shape)
        
        total_size = array.size
        processed = 0
        
        if array.ndim == 1:
            for i in range(0, array.size, chunk_size):
                end_idx = min(i + chunk_size, array.size)
                out[i:end_idx] = array[i:end_idx]
                
                processed += (end_idx - i)
                if progress_callback:
                    progress_callback(processed / total_size)
        else:
            array_flat = array.ravel()
            out_flat = out.ravel()
            
            for i in range(0, array.size, chunk_size):
                end_idx = min(i + chunk_size, array.size)
                out_flat[i:end_idx] = array_flat[i:end_idx]
                
                processed += (end_idx - i)
                if progress_callback:
                    progress_callback(processed / total_size)
        
        out.flush()
        return out
        
    except Exception as e:
        if temp_file is not None and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
        raise e


def chunked_copy_context_manager(array, chunk_size=None):
    """
    Context manager version for automatic cleanup.
    
    Usage:
        with chunked_copy_context_manager(large_array) as copied_array:
            # Use copied_array
            pass
        # Temporary file is automatically cleaned up
    """
    class ChunkedCopyContext:
        def __init__(self, array, chunk_size):
            self.array = array
            self.chunk_size = chunk_size
            self.result = None
            self.temp_file_path = None
            
        def __enter__(self):
            self.result = chunked_copy(self.array, self.chunk_size)
            if hasattr(self.result, 'filename'):
                self.temp_file_path = self.result.filename
            return self.result
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                try:
                    os.unlink(self.temp_file_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    return ChunkedCopyContext(array, chunk_size)


# Example usage and testing
if __name__ == "__main__":
    # Test with different array sizes
    print("Testing chunked_copy function...")
    
    # Small array test
    small_array = np.random.rand(100)
    small_copy = chunked_copy(small_array)
    print(f"Small array test passed: {np.array_equal(small_array, small_copy)}")
    
    # Large array test
    large_array = np.random.rand(5000000)  # ~38MB for float64
    large_copy = chunked_copy(large_array, chunk_size=1000000)
    print(f"Large array test passed: {np.array_equal(large_array, large_copy)}")
    
    # Multi-dimensional array test
    multi_array = np.random.rand(1000, 1000, 5)  # ~38MB for float64
    multi_copy = chunked_copy(multi_array)
    print(f"Multi-dimensional array test passed: {np.array_equal(multi_array, multi_copy)}")
    
    # Progress callback test
    def progress_print(progress):
        print(f"Progress: {progress*100:.1f}%")
    
    print("\nTesting with progress callback:")
    progress_copy = chunked_copy_with_progress(large_array, progress_callback=progress_print)
    print(f"Progress copy test passed: {np.array_equal(large_array, progress_copy)}")
    
    # Context manager test
    print("\nTesting context manager:")
    with chunked_copy_context_manager(multi_array) as ctx_copy:
        print(f"Context manager test passed: {np.array_equal(multi_array, ctx_copy)}")
    
    print("All tests completed successfully!")