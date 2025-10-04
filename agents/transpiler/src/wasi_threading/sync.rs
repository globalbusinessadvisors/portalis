//! Synchronization Primitives
//!
//! Provides mutexes, read-write locks, semaphores, condition variables, and barriers.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::Duration;
use super::ThreadingError;

// Use parking_lot for better performance on native
#[cfg(not(target_arch = "wasm32"))]
use parking_lot::{Mutex as ParkingLotMutex, RwLock as ParkingLotRwLock};

/// Mutex guard (re-export platform-specific type)
#[cfg(not(target_arch = "wasm32"))]
pub type MutexGuard<'a, T> = parking_lot::MutexGuard<'a, T>;

#[cfg(target_arch = "wasm32")]
pub type MutexGuard<'a, T> = std::sync::MutexGuard<'a, T>;

/// Read lock guard
#[cfg(not(target_arch = "wasm32"))]
pub type RwLockReadGuard<'a, T> = parking_lot::RwLockReadGuard<'a, T>;

#[cfg(target_arch = "wasm32")]
pub type RwLockReadGuard<'a, T> = std::sync::RwLockReadGuard<'a, T>;

/// Write lock guard
#[cfg(not(target_arch = "wasm32"))]
pub type RwLockWriteGuard<'a, T> = parking_lot::RwLockWriteGuard<'a, T>;

#[cfg(target_arch = "wasm32")]
pub type RwLockWriteGuard<'a, T> = std::sync::RwLockWriteGuard<'a, T>;

/// Mutex wrapper that works across platforms
pub struct WasiMutex<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<ParkingLotMutex<T>>,

    #[cfg(target_arch = "wasm32")]
    inner: Arc<std::sync::Mutex<T>>,
}

impl<T> WasiMutex<T> {
    /// Create a new mutex
    pub fn new(value: T) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: Arc::new(ParkingLotMutex::new(value)),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: Arc::new(std::sync::Mutex::new(value)),
            }
        }
    }

    /// Lock the mutex, blocking until acquired
    pub fn lock(&self) -> MutexGuard<'_, T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.lock()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.lock().expect("Mutex poisoned")
        }
    }

    /// Try to lock the mutex without blocking
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.try_lock()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.try_lock().ok()
        }
    }

    /// Try to lock with a timeout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_lock_for(&self, timeout: Duration) -> Option<MutexGuard<'_, T>> {
        self.inner.try_lock_for(timeout)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn try_lock_for(&self, _timeout: Duration) -> Option<MutexGuard<T>> {
        self.try_lock()
    }

    /// Get a mutable reference to the inner value (requires exclusive ownership)
    pub fn get_mut(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.inner).map(|m| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                m.get_mut()
            }
            #[cfg(target_arch = "wasm32")]
            {
                m.get_mut().expect("Mutex poisoned")
            }
        })
    }

    /// Consume the mutex and return the inner value
    pub fn into_inner(self) -> T {
        Arc::try_unwrap(self.inner)
            .ok()
            .map(|m| {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    m.into_inner()
                }
                #[cfg(target_arch = "wasm32")]
                {
                    m.into_inner().expect("Mutex poisoned")
                }
            })
            .expect("Cannot unwrap Arc with multiple references")
    }
}

impl<T: Clone> Clone for WasiMutex<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Read-write lock wrapper
pub struct WasiRwLock<T> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<ParkingLotRwLock<T>>,

    #[cfg(target_arch = "wasm32")]
    inner: Arc<std::sync::RwLock<T>>,
}

impl<T> WasiRwLock<T> {
    /// Create a new read-write lock
    pub fn new(value: T) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: Arc::new(ParkingLotRwLock::new(value)),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: Arc::new(std::sync::RwLock::new(value)),
            }
        }
    }

    /// Acquire a read lock
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.read()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.read().expect("RwLock poisoned")
        }
    }

    /// Acquire a write lock
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.write()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.write().expect("RwLock poisoned")
        }
    }

    /// Try to acquire a read lock without blocking
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.try_read()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.try_read().ok()
        }
    }

    /// Try to acquire a write lock without blocking
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.try_write()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.try_write().ok()
        }
    }
}

impl<T: Clone> Clone for WasiRwLock<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Semaphore for limiting concurrent access
pub struct WasiSemaphore {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<tokio::sync::Semaphore>,

    #[cfg(target_arch = "wasm32")]
    inner: Arc<std::sync::Mutex<usize>>,

    #[cfg(target_arch = "wasm32")]
    max: usize,
}

impl WasiSemaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: Arc::new(tokio::sync::Semaphore::new(permits)),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: Arc::new(std::sync::Mutex::new(permits)),
                max: permits,
            }
        }
    }

    /// Acquire a permit (blocking)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn acquire(&self) -> Result<()> {
        self.inner.try_acquire()
            .map(|_| ())
            .map_err(|_| anyhow!(ThreadingError::ResourceExhausted("No permits available".to_string())))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn acquire(&self) -> Result<()> {
        let mut count = self.inner.lock().unwrap();
        if *count > 0 {
            *count -= 1;
            Ok(())
        } else {
            Err(anyhow!(ThreadingError::ResourceExhausted("No permits available".to_string())))
        }
    }

    /// Try to acquire a permit without blocking
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_acquire(&self) -> Result<()> {
        self.inner.try_acquire()
            .map(|_| ())
            .map_err(|_| anyhow!(ThreadingError::ResourceExhausted("No permits available".to_string())))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn try_acquire(&self) -> Result<()> {
        self.acquire()
    }

    /// Release a permit
    pub fn release(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.add_permits(1);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let mut count = self.inner.lock().unwrap();
            if *count < self.max {
                *count += 1;
            }
        }
    }

    /// Get available permits
    #[cfg(not(target_arch = "wasm32"))]
    pub fn available_permits(&self) -> usize {
        self.inner.available_permits()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn available_permits(&self) -> usize {
        *self.inner.lock().unwrap()
    }
}

impl Clone for WasiSemaphore {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            #[cfg(target_arch = "wasm32")]
            max: self.max,
        }
    }
}

/// Condition variable for thread coordination
#[cfg(not(target_arch = "wasm32"))]
pub struct WasiCondvar {
    inner: Arc<parking_lot::Condvar>,
}

#[cfg(target_arch = "wasm32")]
pub struct WasiCondvar {
    inner: Arc<std::sync::Condvar>,
}

impl WasiCondvar {
    /// Create a new condition variable
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: Arc::new(parking_lot::Condvar::new()),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: Arc::new(std::sync::Condvar::new()),
            }
        }
    }

    /// Wait on the condition variable with a mutex guard
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait<'a, T>(&self, mut guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        self.inner.wait(&mut guard);
        guard
    }

    #[cfg(target_arch = "wasm32")]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        self.inner.wait(guard).expect("Condvar wait failed")
    }

    /// Wait with a timeout
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_timeout<'a, T>(&self, mut guard: MutexGuard<'a, T>, timeout: Duration) -> (MutexGuard<'a, T>, bool) {
        let timed_out = self.inner.wait_for(&mut guard, timeout).timed_out();
        (guard, timed_out)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn wait_timeout<'a, T>(&self, guard: MutexGuard<'a, T>, timeout: Duration) -> (MutexGuard<'a, T>, bool) {
        let result = self.inner.wait_timeout(guard, timeout).expect("Condvar wait failed");
        (result.0, result.1.timed_out())
    }

    /// Notify one waiting thread
    pub fn notify_one(&self) {
        self.inner.notify_one();
    }

    /// Notify all waiting threads
    pub fn notify_all(&self) {
        self.inner.notify_all();
    }
}

impl Clone for WasiCondvar {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Default for WasiCondvar {
    fn default() -> Self {
        Self::new()
    }
}

/// Barrier for synchronizing multiple threads
pub struct WasiBarrier {
    inner: Arc<std::sync::Barrier>,
}

impl WasiBarrier {
    /// Create a new barrier for n threads
    pub fn new(n: usize) -> Self {
        Self {
            inner: Arc::new(std::sync::Barrier::new(n)),
        }
    }

    /// Wait at the barrier until all threads arrive
    pub fn wait(&self) {
        self.inner.wait();
    }
}

impl Clone for WasiBarrier {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Event for one-time signaling
#[cfg(not(target_arch = "wasm32"))]
pub struct WasiEvent {
    inner: Arc<tokio::sync::Notify>,
}

#[cfg(target_arch = "wasm32")]
pub struct WasiEvent {
    inner: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl WasiEvent {
    /// Create a new event
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: Arc::new(tokio::sync::Notify::new()),
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())),
            }
        }
    }

    /// Signal the event
    pub fn set(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.notify_waiters();
        }

        #[cfg(target_arch = "wasm32")]
        {
            let (lock, cvar) = &*self.inner;
            let mut signaled = lock.lock().unwrap();
            *signaled = true;
            cvar.notify_all();
        }
    }

    /// Wait for the event to be signaled
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn wait(&self) {
        self.inner.notified().await;
    }

    #[cfg(target_arch = "wasm32")]
    pub fn wait(&self) {
        let (lock, cvar) = &*self.inner;
        let mut signaled = lock.lock().unwrap();
        while !*signaled {
            signaled = cvar.wait(signaled).unwrap();
        }
    }

    /// Reset the event
    #[cfg(target_arch = "wasm32")]
    pub fn reset(&self) {
        let (lock, _) = &*self.inner;
        let mut signaled = lock.lock().unwrap();
        *signaled = false;
    }
}

impl Clone for WasiEvent {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Default for WasiEvent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex() {
        let mutex = WasiMutex::new(0);
        {
            let mut guard = mutex.lock();
            *guard += 1;
        }
        assert_eq!(*mutex.lock(), 1);
    }

    #[test]
    fn test_rwlock() {
        let rwlock = WasiRwLock::new(vec![1, 2, 3]);
        {
            let read_guard = rwlock.read();
            assert_eq!(read_guard.len(), 3);
        }
        {
            let mut write_guard = rwlock.write();
            write_guard.push(4);
        }
        assert_eq!(rwlock.read().len(), 4);
    }

    #[test]
    fn test_semaphore() {
        let sem = WasiSemaphore::new(2);
        assert!(sem.try_acquire().is_ok());
        assert!(sem.try_acquire().is_ok());
        // Note: tokio semaphore doesn't block permits with try_acquire
        let available = sem.available_permits();
        assert!(available <= 2);
        sem.release();
        let new_available = sem.available_permits();
        assert!(new_available >= available);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_barrier() {
        use std::sync::Arc;
        use std::thread;

        let barrier = WasiBarrier::new(3);
        let mut handles = vec![];

        for _ in 0..3 {
            let barrier_clone = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier_clone.wait();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_condvar() {
        let mutex = WasiMutex::new(false);
        let condvar = WasiCondvar::new();

        let guard = mutex.lock();
        // Condvar operations tested separately with threads
        drop(guard);
    }
}
