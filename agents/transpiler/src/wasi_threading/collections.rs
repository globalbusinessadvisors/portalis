//! Thread-Safe Collections
//!
//! Provides concurrent queues, stacks, priority queues, and deques.

use anyhow::{Result, anyhow};
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;
use super::{WasiMutex, WasiCondvar, ThreadingError};

#[cfg(not(target_arch = "wasm32"))]
use crossbeam::channel::{Sender, Receiver, bounded, unbounded};

/// Thread-safe queue (FIFO)
pub struct WasiQueue<T> {
    #[cfg(not(target_arch = "wasm32"))]
    sender: Sender<T>,

    #[cfg(not(target_arch = "wasm32"))]
    receiver: Receiver<T>,

    #[cfg(target_arch = "wasm32")]
    inner: WasiMutex<VecDeque<T>>,

    #[cfg(target_arch = "wasm32")]
    condvar: WasiCondvar,
}

impl<T> WasiQueue<T> {
    /// Create a new unbounded queue
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = unbounded();
            Self { sender, receiver }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: WasiMutex::new(VecDeque::new()),
                condvar: WasiCondvar::new(),
            }
        }
    }

    /// Create a new bounded queue with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = bounded(capacity);
            Self { sender, receiver }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self {
                inner: WasiMutex::new(VecDeque::with_capacity(capacity)),
                condvar: WasiCondvar::new(),
            }
        }
    }

    /// Push an item to the back of the queue
    pub fn push(&self, item: T) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sender.send(item)
                .map_err(|_| anyhow!(ThreadingError::ChannelClosed("Queue is closed".to_string())))
        }

        #[cfg(target_arch = "wasm32")]
        {
            let mut queue = self.inner.lock();
            queue.push_back(item);
            self.condvar.notify_one();
            Ok(())
        }
    }

    /// Try to push an item without blocking (for bounded queues)
    pub fn try_push(&self, item: T) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sender.try_send(item)
                .map_err(|e| match e {
                    crossbeam::channel::TrySendError::Full(_) => {
                        anyhow!(ThreadingError::ResourceExhausted("Queue is full".to_string()))
                    }
                    crossbeam::channel::TrySendError::Disconnected(_) => {
                        anyhow!(ThreadingError::ChannelClosed("Queue is closed".to_string()))
                    }
                })
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.push(item)
        }
    }

    /// Pop an item from the front of the queue (blocking)
    pub fn pop(&self) -> Result<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.recv()
                .map_err(|_| anyhow!(ThreadingError::ChannelClosed("Queue is closed".to_string())))
        }

        #[cfg(target_arch = "wasm32")]
        {
            let mut queue = self.inner.lock();
            while queue.is_empty() {
                queue = self.condvar.wait(queue);
            }
            queue.pop_front()
                .ok_or_else(|| anyhow!(ThreadingError::ChannelClosed("Queue is empty".to_string())))
        }
    }

    /// Try to pop an item without blocking
    pub fn try_pop(&self) -> Option<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.try_recv().ok()
        }

        #[cfg(target_arch = "wasm32")]
        {
            let mut queue = self.inner.lock();
            queue.pop_front()
        }
    }

    /// Get the current length of the queue
    pub fn len(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.len()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.inner.lock().len()
        }
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for WasiQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe stack (LIFO)
pub struct WasiStack<T> {
    inner: WasiMutex<Vec<T>>,
    condvar: WasiCondvar,
}

impl<T> WasiStack<T> {
    /// Create a new stack
    pub fn new() -> Self {
        Self {
            inner: WasiMutex::new(Vec::new()),
            condvar: WasiCondvar::new(),
        }
    }

    /// Create a new stack with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: WasiMutex::new(Vec::with_capacity(capacity)),
            condvar: WasiCondvar::new(),
        }
    }

    /// Push an item onto the stack
    pub fn push(&self, item: T) {
        let mut stack = self.inner.lock();
        stack.push(item);
        self.condvar.notify_one();
    }

    /// Pop an item from the stack (blocking if empty)
    pub fn pop(&self) -> Result<T> {
        let mut stack = self.inner.lock();
        while stack.is_empty() {
            stack = self.condvar.wait(stack);
        }
        stack.pop()
            .ok_or_else(|| anyhow!(ThreadingError::ChannelClosed("Stack is empty".to_string())))
    }

    /// Try to pop an item without blocking
    pub fn try_pop(&self) -> Option<T> {
        let mut stack = self.inner.lock();
        stack.pop()
    }

    /// Get the current length of the stack
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Peek at the top item without removing it
    pub fn peek(&self) -> Option<T>
    where
        T: Clone,
    {
        self.inner.lock().last().cloned()
    }
}

impl<T> Default for WasiStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority queue item wrapper
#[derive(Debug, Clone)]
struct PriorityItem<T, P> {
    item: T,
    priority: P,
}

impl<T, P: Ord> PartialEq for PriorityItem<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T, P: Ord> Eq for PriorityItem<T, P> {}

impl<T, P: Ord> PartialOrd for PriorityItem<T, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, P: Ord> Ord for PriorityItem<T, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

/// Thread-safe priority queue (max-heap by default)
pub struct WasiPriorityQueue<T, P = i32>
where
    P: Ord,
{
    inner: WasiMutex<BinaryHeap<PriorityItem<T, P>>>,
    condvar: WasiCondvar,
}

impl<T, P: Ord> WasiPriorityQueue<T, P> {
    /// Create a new priority queue
    pub fn new() -> Self {
        Self {
            inner: WasiMutex::new(BinaryHeap::new()),
            condvar: WasiCondvar::new(),
        }
    }

    /// Create a new priority queue with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: WasiMutex::new(BinaryHeap::with_capacity(capacity)),
            condvar: WasiCondvar::new(),
        }
    }

    /// Push an item with the given priority
    pub fn push(&self, item: T, priority: P) {
        let mut heap = self.inner.lock();
        heap.push(PriorityItem { item, priority });
        self.condvar.notify_one();
    }

    /// Pop the highest priority item (blocking if empty)
    pub fn pop(&self) -> Result<T> {
        let mut heap = self.inner.lock();
        while heap.is_empty() {
            heap = self.condvar.wait(heap);
        }
        heap.pop()
            .map(|item| item.item)
            .ok_or_else(|| anyhow!(ThreadingError::ChannelClosed("Queue is empty".to_string())))
    }

    /// Try to pop the highest priority item without blocking
    pub fn try_pop(&self) -> Option<T> {
        let mut heap = self.inner.lock();
        heap.pop().map(|item| item.item)
    }

    /// Peek at the highest priority item without removing it
    pub fn peek(&self) -> Option<(T, P)>
    where
        T: Clone,
        P: Clone,
    {
        let heap = self.inner.lock();
        heap.peek().map(|item| (item.item.clone(), item.priority.clone()))
    }

    /// Get the current length of the queue
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, P: Ord> Default for WasiPriorityQueue<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe double-ended queue
pub struct WasiDeque<T> {
    inner: WasiMutex<VecDeque<T>>,
    condvar: WasiCondvar,
}

impl<T> WasiDeque<T> {
    /// Create a new deque
    pub fn new() -> Self {
        Self {
            inner: WasiMutex::new(VecDeque::new()),
            condvar: WasiCondvar::new(),
        }
    }

    /// Create a new deque with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: WasiMutex::new(VecDeque::with_capacity(capacity)),
            condvar: WasiCondvar::new(),
        }
    }

    /// Push an item to the back
    pub fn push_back(&self, item: T) {
        let mut deque = self.inner.lock();
        deque.push_back(item);
        self.condvar.notify_one();
    }

    /// Push an item to the front
    pub fn push_front(&self, item: T) {
        let mut deque = self.inner.lock();
        deque.push_front(item);
        self.condvar.notify_one();
    }

    /// Pop an item from the back (blocking if empty)
    pub fn pop_back(&self) -> Result<T> {
        let mut deque = self.inner.lock();
        while deque.is_empty() {
            deque = self.condvar.wait(deque);
        }
        deque.pop_back()
            .ok_or_else(|| anyhow!(ThreadingError::ChannelClosed("Deque is empty".to_string())))
    }

    /// Pop an item from the front (blocking if empty)
    pub fn pop_front(&self) -> Result<T> {
        let mut deque = self.inner.lock();
        while deque.is_empty() {
            deque = self.condvar.wait(deque);
        }
        deque.pop_front()
            .ok_or_else(|| anyhow!(ThreadingError::ChannelClosed("Deque is empty".to_string())))
    }

    /// Try to pop an item from the back without blocking
    pub fn try_pop_back(&self) -> Option<T> {
        let mut deque = self.inner.lock();
        deque.pop_back()
    }

    /// Try to pop an item from the front without blocking
    pub fn try_pop_front(&self) -> Option<T> {
        let mut deque = self.inner.lock();
        deque.pop_front()
    }

    /// Get the current length
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Check if the deque is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for WasiDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue() {
        let queue = WasiQueue::new();
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.try_pop(), Some(1));
        assert_eq!(queue.try_pop(), Some(2));
        assert_eq!(queue.try_pop(), Some(3));
        assert_eq!(queue.try_pop(), None);
    }

    #[test]
    fn test_stack() {
        let stack = WasiStack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.try_pop(), Some(3));
        assert_eq!(stack.try_pop(), Some(2));
        assert_eq!(stack.try_pop(), Some(1));
        assert_eq!(stack.try_pop(), None);
    }

    #[test]
    fn test_priority_queue() {
        let pq = WasiPriorityQueue::new();
        pq.push("low", 1);
        pq.push("high", 10);
        pq.push("medium", 5);

        assert_eq!(pq.len(), 3);
        assert_eq!(pq.try_pop(), Some("high"));
        assert_eq!(pq.try_pop(), Some("medium"));
        assert_eq!(pq.try_pop(), Some("low"));
        assert_eq!(pq.try_pop(), None);
    }

    #[test]
    fn test_deque() {
        let deque = WasiDeque::new();
        deque.push_back(1);
        deque.push_front(2);
        deque.push_back(3);

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.try_pop_front(), Some(2));
        assert_eq!(deque.try_pop_back(), Some(3));
        assert_eq!(deque.try_pop_front(), Some(1));
        assert_eq!(deque.try_pop_back(), None);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_queue_concurrent() {
        use std::thread;
        use std::sync::Arc;

        let queue = Arc::new(WasiQueue::new());
        let queue_clone = queue.clone();

        let producer = thread::spawn(move || {
            for i in 0..10 {
                queue_clone.push(i).unwrap();
            }
        });

        let consumer = thread::spawn(move || {
            let mut sum = 0;
            for _ in 0..10 {
                sum += queue.pop().unwrap();
            }
            sum
        });

        producer.join().unwrap();
        let sum = consumer.join().unwrap();
        assert_eq!(sum, 45); // 0+1+2+...+9 = 45
    }
}
