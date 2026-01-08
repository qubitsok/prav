//! Arena-based bump allocator for `no_std` environments.
//!
//! This module provides a simple, fast bump allocator that works without heap allocation.
//! The arena pre-allocates a fixed buffer and hands out memory from it sequentially.
//!
//! # Why Use an Arena?
//!
//! In quantum error correction decoding, we need to allocate many data structures
//! (parent arrays, block states, masks) but:
//!
//! - We know the maximum size upfront (determined by grid dimensions)
//! - We want predictable, cache-friendly memory layout
//! - We need `no_std` compatibility for embedded/FPGA targets
//! - Traditional allocators add overhead we don't need
//!
//! # Usage Pattern
//!
//! ```ignore
//! // Calculate required buffer size
//! let size = required_buffer_size(32, 32, 1);
//!
//! // Pre-allocate buffer (on stack or static)
//! let mut buffer = [0u8; size];
//! let mut arena = Arena::new(&mut buffer);
//!
//! // Allocate decoder data structures
//! let parents = arena.alloc_slice::<u32>(num_nodes)?;
//! let blocks = arena.alloc_slice_aligned::<BlockStateHot>(num_blocks, 64)?;
//!
//! // After decoding cycle, optionally reset for reuse
//! arena.reset();
//! ```

use core::mem::{align_of, size_of};

/// Calculates the required buffer size for a QEC decoder.
///
/// This function computes a conservative upper bound on the memory needed
/// for all internal data structures, including alignment padding.
///
/// # Arguments
///
/// * `width` - Grid width in nodes.
/// * `height` - Grid height in nodes.
/// * `depth` - Grid depth (1 for 2D codes, >1 for 3D codes).
///
/// # Returns
///
/// The minimum buffer size in bytes required for [`Arena::new`].
///
/// # Example
///
/// ```
/// use prav_core::required_buffer_size;
///
/// // Calculate buffer for 32x32 2D grid
/// let size = required_buffer_size(32, 32, 1);
/// assert!(size > 0);
///
/// // Larger grids need more memory
/// let size_64 = required_buffer_size(64, 64, 1);
/// assert!(size_64 > size);
/// ```
#[must_use]
pub const fn required_buffer_size(width: usize, height: usize, depth: usize) -> usize {
    let is_3d = depth > 1;
    let max_dim = const_max(width, const_max(height, if is_3d { depth } else { 1 }));
    let dim_pow2 = max_dim.next_power_of_two();

    let alloc_size = if is_3d {
        dim_pow2 * dim_pow2 * dim_pow2
    } else {
        dim_pow2 * dim_pow2
    };
    let alloc_nodes = alloc_size + 1;
    let num_blocks = div_ceil(alloc_nodes, 64);
    let num_bitmask_words = div_ceil(num_blocks, 64);
    let num_edges = alloc_nodes * 3;
    let num_edge_words = div_ceil(num_edges, 64);

    // Each allocation has up to 63 bytes of alignment padding (for 64-byte alignment)
    const ALIGN_PAD: usize = 64;

    let mut total = 0;

    // StaticGraph (~128 bytes)
    total += 128 + ALIGN_PAD;

    // blocks_state: num_blocks * 64 bytes (BlockStateHot is 64 bytes)
    total += num_blocks * 64 + ALIGN_PAD;

    // parents: alloc_nodes * 4 bytes (u32)
    total += alloc_nodes * 4 + ALIGN_PAD;

    // defect_mask: num_blocks * 8 bytes (u64)
    total += num_blocks * 8 + ALIGN_PAD;

    // path_mark: num_blocks * 8 bytes (u64)
    total += num_blocks * 8 + ALIGN_PAD;

    // block_dirty_mask: div_ceil(num_blocks, 64) * 8 bytes
    total += div_ceil(num_blocks, 64) * 8 + ALIGN_PAD;

    // active_mask: num_bitmask_words * 8 bytes
    total += num_bitmask_words * 8 + ALIGN_PAD;

    // queued_mask: num_bitmask_words * 8 bytes
    total += num_bitmask_words * 8 + ALIGN_PAD;

    // ingestion_list: num_blocks * 4 bytes (u32)
    total += num_blocks * 4 + ALIGN_PAD;

    // edge_bitmap: num_edge_words * 8 bytes
    total += num_edge_words * 8 + ALIGN_PAD;

    // edge_dirty_list: num_edge_words * 8 * 4 bytes (u32, 8x overalloc)
    total += num_edge_words * 8 * 4 + ALIGN_PAD;

    // boundary_bitmap: num_blocks * 8 bytes
    total += num_blocks * 8 + ALIGN_PAD;

    // boundary_dirty_list: num_blocks * 8 * 4 bytes (u32, 8x overalloc)
    total += num_blocks * 8 * 4 + ALIGN_PAD;

    // edge_dirty_mask: div_ceil(num_edge_words, 64) * 8 bytes
    total += div_ceil(num_edge_words, 64) * 8 + ALIGN_PAD;

    // boundary_dirty_mask: div_ceil(num_blocks, 64) * 8 bytes
    total += div_ceil(num_blocks, 64) * 8 + ALIGN_PAD;

    // bfs_pred: alloc_nodes * 2 bytes (u16)
    total += alloc_nodes * 2 + ALIGN_PAD;

    // bfs_queue: alloc_nodes * 2 bytes (u16)
    total += alloc_nodes * 2 + ALIGN_PAD;

    total
}

/// Const-compatible max function.
const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// Const-compatible ceiling division.
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// A bump allocator that manages a pre-allocated byte buffer.
///
/// The arena allocates memory sequentially from a fixed buffer, making allocation
/// O(1) and deallocation trivial (just reset the offset). This is ideal for
/// QEC decoding where all memory is allocated upfront and freed together.
///
/// # Memory Layout
///
/// ```text
/// Buffer: [====allocated====|----available----|]
///         ^                 ^                  ^
///         base              offset             end
/// ```
///
/// Each allocation advances the offset, with padding added for alignment.
///
/// # Thread Safety
///
/// The arena is not thread-safe. For parallel decoding, use one arena per thread.
pub struct Arena<'a> {
    /// The underlying byte buffer from which memory is allocated.
    buffer: &'a mut [u8],
    /// Current allocation offset within the buffer.
    offset: usize,
}

impl<'a> Arena<'a> {
    /// Creates a new arena backed by the given buffer.
    ///
    /// The buffer should be large enough to hold all decoder data structures.
    /// A typical formula for required size:
    ///
    /// ```text
    /// size ≈ num_nodes * 4 (parents)
    ///      + num_blocks * 64 (BlockStateHot)
    ///      + num_blocks * 8 (masks)
    ///      + overhead for alignment padding
    /// ```
    ///
    /// # Arguments
    ///
    /// * `buffer` - Mutable byte slice to use for allocations. Can be stack-allocated,
    ///   static, or from any other source.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Stack allocation (be careful with stack size limits)
    /// let mut buffer = [0u8; 64 * 1024];
    /// let mut arena = Arena::new(&mut buffer);
    ///
    /// // Static allocation (good for embedded)
    /// static mut BUFFER: [u8; 1024 * 1024] = [0; 1024 * 1024];
    /// let arena = unsafe { Arena::new(&mut BUFFER) };
    /// ```
    #[must_use]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer, offset: 0 }
    }

    /// Allocates a slice of `len` elements with natural alignment for type `T`.
    ///
    /// This is the most common allocation method. It ensures the returned slice
    /// is properly aligned for type `T`.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate.
    ///
    /// # Returns
    ///
    /// * `Ok(&mut [T])` - Mutable slice of uninitialized memory. The caller is
    ///   responsible for initializing elements before reading them.
    /// * `Err(&'static str)` - If the arena doesn't have enough space.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allocate parent array for Union Find
    /// let parents: &mut [u32] = arena.alloc_slice(num_nodes)?;
    ///
    /// // Initialize: each node is its own parent (self-rooted)
    /// for i in 0..num_nodes {
    ///     parents[i] = i as u32;
    /// }
    /// ```
    #[inline]
    pub fn alloc_slice<T: Copy>(&mut self, len: usize) -> Result<&'a mut [T], &'static str> {
        self.alloc_slice_aligned(len, align_of::<T>())
    }

    /// Allocates a slice with custom alignment, useful for cache-line alignment.
    ///
    /// In QEC decoding, cache efficiency is critical. The `BlockStateHot` struct
    /// is designed to fit exactly in a 64-byte cache line, so we allocate it
    /// with 64-byte alignment to prevent false sharing and ensure optimal access.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate.
    /// * `align` - Desired alignment in bytes. If less than `T`'s natural alignment,
    ///   the natural alignment is used instead.
    ///
    /// # Returns
    ///
    /// * `Ok(&mut [T])` - Mutable slice with the requested alignment.
    /// * `Err(&'static str)` - If the arena doesn't have enough space.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allocate BlockStateHot array with cache-line alignment
    /// let blocks: &mut [BlockStateHot] = arena.alloc_slice_aligned(num_blocks, 64)?;
    /// ```
    ///
    /// # Alignment Calculation
    ///
    /// The method calculates padding to reach the requested alignment:
    ///
    /// ```text
    /// current_ptr = base + offset
    /// padding = (align - (current_ptr % align)) % align
    /// new_offset = offset + padding
    /// ```
    #[inline]
    pub fn alloc_slice_aligned<T: Copy>(
        &mut self,
        len: usize,
        align: usize,
    ) -> Result<&'a mut [T], &'static str> {
        let t_align = align_of::<T>();
        let actual_align = if align > t_align { align } else { t_align };
        let size = size_of::<T>() * len;

        // Get the actual memory address
        let base_ptr = self.buffer.as_mut_ptr() as usize;
        let current_ptr = base_ptr + self.offset;

        // Calculate padding required to align the actual address
        let padding = (actual_align - (current_ptr % actual_align)) % actual_align;

        if self.offset + padding + size > self.buffer.len() {
            return Err("OOM: Arena too small");
        }

        self.offset += padding;
        // SAFETY: We've verified that offset + size <= buffer.len() above.
        // The pointer arithmetic stays within the buffer bounds.
        let ptr = unsafe { self.buffer.as_mut_ptr().add(self.offset) as *mut T };
        self.offset += size;

        // SAFETY: ptr is properly aligned (we added padding for alignment),
        // points to at least `len * size_of::<T>()` bytes of valid memory,
        // and the lifetime is tied to the arena's buffer lifetime 'a.
        unsafe { Ok(core::slice::from_raw_parts_mut(ptr, len)) }
    }

    /// Allocates a single value with natural alignment.
    ///
    /// This is a convenience method for allocating single values rather than slices.
    /// The value is copied into the arena and a mutable reference is returned.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store in the arena.
    ///
    /// # Returns
    ///
    /// * `Ok(&mut T)` - Mutable reference to the stored value.
    /// * `Err(&'static str)` - If the arena doesn't have enough space.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let graph: &mut StaticGraph = arena.alloc_value(StaticGraph {
    ///     width: 32,
    ///     height: 32,
    ///     // ...
    /// })?;
    /// ```
    #[inline]
    pub fn alloc_value<T: Copy>(&mut self, value: T) -> Result<&'a mut T, &'static str> {
        let slice = self.alloc_slice_aligned(1, align_of::<T>())?;
        slice[0] = value;
        Ok(&mut slice[0])
    }

    /// Resets the arena, allowing all memory to be reused.
    ///
    /// This simply sets the offset back to zero. Previously allocated slices
    /// become invalid (dangling), so the caller must ensure they are not used
    /// after reset.
    ///
    /// # Use Case
    ///
    /// In QEC decoding pipelines, you might want to reuse the same arena
    /// across multiple decoding cycles rather than allocating new memory
    /// each time.
    ///
    /// # Safety Note
    ///
    /// After calling `reset()`, all previously allocated references become
    /// invalid. Using them is undefined behavior. The borrow checker should
    /// prevent this in safe code, but be careful with raw pointers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for cycle in 0..num_cycles {
    ///     arena.reset();
    ///     let state = DecodingState::new(&mut arena, width, height, depth);
    ///     // ... decode cycle ...
    /// }
    /// ```
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
