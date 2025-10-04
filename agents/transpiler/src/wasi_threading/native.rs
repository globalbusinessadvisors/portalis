//! Native Platform Implementation
//!
//! Native implementation using std::thread, rayon, and parking_lot.
//! This provides the best performance on native platforms.

use std::sync::Arc;
use parking_lot::{Mutex, RwLock};

/// Native mutex implementation using parking_lot
#[allow(dead_code)]
pub type NativeMutex<T> = Arc<Mutex<T>>;

/// Native RwLock implementation using parking_lot
#[allow(dead_code)]
pub type NativeRwLock<T> = Arc<RwLock<T>>;

/// Platform-specific thread utilities
#[allow(dead_code)]
pub struct NativeThreadUtils;

#[allow(dead_code)]
impl NativeThreadUtils {
    /// Set thread priority (platform-specific)
    #[cfg(target_os = "linux")]
    pub fn set_priority(priority: super::ThreadPriority) -> std::io::Result<()> {
        use libc::{pthread_self, pthread_setschedprio};

        let priority_value = match priority {
            super::ThreadPriority::Low => -10,
            super::ThreadPriority::Normal => 0,
            super::ThreadPriority::High => 10,
        };

        unsafe {
            let result = pthread_setschedprio(pthread_self(), priority_value);
            if result == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn set_priority(_priority: super::ThreadPriority) -> std::io::Result<()> {
        // Not supported on this platform
        Ok(())
    }

    /// Pin thread to specific CPU core
    #[cfg(target_os = "linux")]
    pub fn pin_to_core(core_id: usize) -> std::io::Result<()> {
        use libc::{pthread_self, cpu_set_t, CPU_SET, pthread_setaffinity_np};
        use std::mem;

        unsafe {
            let mut cpuset: cpu_set_t = mem::zeroed();
            CPU_SET(core_id, &mut cpuset);

            let result = pthread_setaffinity_np(
                pthread_self(),
                mem::size_of::<cpu_set_t>(),
                &cpuset,
            );

            if result == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn pin_to_core(_core_id: usize) -> std::io::Result<()> {
        // Not supported on this platform
        Ok(())
    }

    /// Get current thread's CPU affinity
    #[cfg(target_os = "linux")]
    pub fn get_affinity() -> std::io::Result<Vec<usize>> {
        use libc::{pthread_self, cpu_set_t, CPU_ISSET, pthread_getaffinity_np};
        use std::mem;

        unsafe {
            let mut cpuset: cpu_set_t = mem::zeroed();

            let result = pthread_getaffinity_np(
                pthread_self(),
                mem::size_of::<cpu_set_t>(),
                &mut cpuset,
            );

            if result != 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut cores = Vec::new();
            for i in 0..num_cpus::get() {
                if CPU_ISSET(i, &cpuset) {
                    cores.push(i);
                }
            }

            Ok(cores)
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn get_affinity() -> std::io::Result<Vec<usize>> {
        // Not supported, return all cores
        Ok((0..num_cpus::get()).collect())
    }
}

/// Thread-local storage implementation
#[macro_export]
macro_rules! native_thread_local {
    ($name:ident: $ty:ty = $init:expr) => {
        thread_local! {
            static $name: std::cell::RefCell<$ty> = std::cell::RefCell::new($init);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_utils() {
        // Test that functions don't panic
        let _ = NativeThreadUtils::set_priority(super::super::ThreadPriority::Normal);
        let _ = NativeThreadUtils::get_affinity();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_affinity() {
        let affinity = NativeThreadUtils::get_affinity().unwrap();
        assert!(!affinity.is_empty());
        assert!(affinity.len() <= num_cpus::get());
    }
}
