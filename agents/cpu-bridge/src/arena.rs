//! Arena allocation for AST nodes and temporary data structures
//!
//! Provides bump allocation for fast, cache-friendly memory allocation
//! during transpilation. All allocations are freed when the arena is dropped.

use bumpalo::Bump;
use std::cell::RefCell;

/// Thread-local arena for AST node allocation
///
/// Uses bump allocation for extremely fast allocation (just pointer bump).
/// All memory is freed when the arena is dropped, eliminating per-node deallocation.
///
/// # Performance
/// - Allocation: ~10-50ns (vs ~100-500ns for malloc)
/// - Deallocation: Free (batch deallocation when arena drops)
/// - Memory overhead: ~1% (vs ~16-32 bytes per allocation for malloc)
pub struct Arena {
    bump: Bump,
    /// Track total bytes allocated
    bytes_allocated: RefCell<usize>,
    /// Track number of allocations
    allocation_count: RefCell<usize>,
}

impl Arena {
    /// Create a new arena with default capacity (4KB)
    pub fn new() -> Self {
        Self::with_capacity(4 * 1024)
    }

    /// Create a new arena with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Initial capacity in bytes
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
            bytes_allocated: RefCell::new(0),
            allocation_count: RefCell::new(0),
        }
    }

    /// Allocate a value in the arena
    ///
    /// Returns a reference with the arena's lifetime.
    pub fn alloc<T>(&self, value: T) -> &T {
        let size = std::mem::size_of::<T>();
        *self.bytes_allocated.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        self.bump.alloc(value)
    }

    /// Allocate a slice in the arena
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        let size = std::mem::size_of::<T>() * slice.len();
        *self.bytes_allocated.borrow_mut() += size;
        *self.allocation_count.borrow_mut() += 1;

        self.bump.alloc_slice_copy(slice)
    }

    /// Allocate a string in the arena
    pub fn alloc_str(&self, s: &str) -> &str {
        *self.bytes_allocated.borrow_mut() += s.len();
        *self.allocation_count.borrow_mut() += 1;

        self.bump.alloc_str(s)
    }

    /// Get statistics about arena usage
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            bytes_allocated: *self.bytes_allocated.borrow(),
            allocation_count: *self.allocation_count.borrow(),
            bytes_capacity: self.bump.allocated_bytes(),
        }
    }

    /// Reset the arena, freeing all allocations
    ///
    /// This is faster than dropping and recreating the arena.
    pub fn reset(&mut self) {
        self.bump.reset();
        *self.bytes_allocated.borrow_mut() = 0;
        *self.allocation_count.borrow_mut() = 0;
    }

    /// Get the current allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about arena usage
#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    /// Total bytes allocated
    pub bytes_allocated: usize,
    /// Number of allocations made
    pub allocation_count: usize,
    /// Current arena capacity
    pub bytes_capacity: usize,
}

impl ArenaStats {
    /// Calculate average allocation size
    pub fn avg_allocation_size(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.bytes_allocated as f64 / self.allocation_count as f64
        }
    }

    /// Calculate memory efficiency (allocated / capacity)
    pub fn efficiency(&self) -> f64 {
        if self.bytes_capacity == 0 {
            0.0
        } else {
            self.bytes_allocated as f64 / self.bytes_capacity as f64
        }
    }
}

/// Arena pool for managing multiple arenas
///
/// Reuses arenas to avoid repeated allocation/deallocation of the arena itself.
pub struct ArenaPool {
    pool: crossbeam::queue::SegQueue<Arena>,
    /// Default arena capacity
    default_capacity: usize,
    /// Maximum pool size
    max_pool_size: usize,
}

impl ArenaPool {
    /// Create a new arena pool
    pub fn new(default_capacity: usize, max_pool_size: usize) -> Self {
        Self {
            pool: crossbeam::queue::SegQueue::new(),
            default_capacity,
            max_pool_size,
        }
    }

    /// Acquire an arena from the pool
    ///
    /// If the pool is empty, creates a new arena.
    pub fn acquire(&self) -> PooledArena {
        let arena = self
            .pool
            .pop()
            .unwrap_or_else(|| Arena::with_capacity(self.default_capacity));

        PooledArena {
            arena: Some(arena),
            pool: self,
        }
    }

    /// Return an arena to the pool
    fn release(&self, arena: Arena) {
        if self.pool.len() < self.max_pool_size {
            self.pool.push(arena);
        }
        // Otherwise drop the arena
    }

    /// Get current pool size
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

/// RAII wrapper for pooled arenas
///
/// Automatically returns the arena to the pool when dropped.
pub struct PooledArena<'a> {
    arena: Option<Arena>,
    pool: &'a ArenaPool,
}

impl<'a> PooledArena<'a> {
    /// Get a reference to the arena
    pub fn get(&self) -> &Arena {
        self.arena.as_ref().unwrap()
    }
}

impl<'a> Drop for PooledArena<'a> {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.take() {
            self.pool.release(arena);
        }
    }
}

impl<'a> std::ops::Deref for PooledArena<'a> {
    type Target = Arena;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let arena = Arena::new();

        let x = arena.alloc(42);
        let y = arena.alloc(100);

        assert_eq!(*x, 42);
        assert_eq!(*y, 100);

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.bytes_allocated >= 8); // At least 2 * sizeof(i32)
    }

    #[test]
    fn test_arena_string() {
        let arena = Arena::new();

        let s1 = arena.alloc_str("hello");
        let s2 = arena.alloc_str("world");

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");

        let stats = arena.stats();
        assert_eq!(stats.bytes_allocated, 10); // 5 + 5 bytes
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new();

        let _x = arena.alloc(42);
        let _y = arena.alloc(100);

        let stats_before = arena.stats();
        assert_eq!(stats_before.allocation_count, 2);

        arena.reset();

        let stats_after = arena.stats();
        assert_eq!(stats_after.allocation_count, 0);
        assert_eq!(stats_after.bytes_allocated, 0);
    }

    #[test]
    fn test_arena_pool() {
        let pool = ArenaPool::new(4096, 10);

        {
            let arena = pool.acquire();
            let _x = arena.alloc(42);
            // Arena returned to pool on drop
        }

        assert!(pool.len() >= 1); // At least one arena in pool

        {
            let arena = pool.acquire();
            // Should reuse the arena from pool
            // Note: Arena is not automatically reset when returned
            assert!(arena.stats().bytes_capacity > 0);
        }
    }

    #[test]
    fn test_arena_stats() {
        let arena = Arena::new();

        arena.alloc(42_i32); // 4 bytes
        arena.alloc(100_i32); // 4 bytes

        let stats = arena.stats();
        assert_eq!(stats.avg_allocation_size(), 4.0);
    }
}
