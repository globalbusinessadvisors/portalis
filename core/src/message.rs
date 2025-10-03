//! Message passing infrastructure for agent communication
//!
//! Following London School TDD: agents communicate via messages, not direct calls.
//! This enables easy mocking and testing of agent interactions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{AgentId, Error, Result};

/// Unique message identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(Uuid);

impl MessageId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Message payload types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Start processing
    Start,
    /// Data payload (JSON-encoded)
    Data(serde_json::Value),
    /// Task completed successfully
    Complete,
    /// Task failed with error
    Error(String),
    /// Request for status
    StatusRequest,
    /// Status response
    StatusResponse(String),
}

/// Message envelope for agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub from: AgentId,
    pub to: AgentId,
    pub payload: MessagePayload,
    pub timestamp: i64,
}

impl Message {
    /// Create a new message
    pub fn new(from: AgentId, to: AgentId, payload: MessagePayload) -> Self {
        Self {
            id: MessageId::new(),
            from,
            to,
            payload,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Create a data message
    pub fn data(from: AgentId, to: AgentId, data: serde_json::Value) -> Self {
        Self::new(from, to, MessagePayload::Data(data))
    }

    /// Create a start message
    pub fn start(from: AgentId, to: AgentId) -> Self {
        Self::new(from, to, MessagePayload::Start)
    }

    /// Create a complete message
    pub fn complete(from: AgentId, to: AgentId) -> Self {
        Self::new(from, to, MessagePayload::Complete)
    }

    /// Create an error message
    pub fn error(from: AgentId, to: AgentId, error: impl Into<String>) -> Self {
        Self::new(from, to, MessagePayload::Error(error.into()))
    }
}

/// Message bus for agent communication
///
/// Following London School TDD:
/// - Agents don't call each other directly
/// - All communication goes through the message bus
/// - Easy to mock for testing
pub struct MessageBus {
    channels: HashMap<AgentId, mpsc::UnboundedSender<Message>>,
}

impl MessageBus {
    /// Create a new message bus
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Register an agent with the message bus
    pub fn register(&mut self, agent_id: AgentId) -> mpsc::UnboundedReceiver<Message> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.channels.insert(agent_id, tx);
        rx
    }

    /// Unregister an agent from the message bus
    pub fn unregister(&mut self, agent_id: &AgentId) {
        self.channels.remove(agent_id);
    }

    /// Send a message to a specific agent
    pub async fn send(&self, message: Message) -> Result<()> {
        let channel = self.channels
            .get(&message.to)
            .ok_or_else(|| Error::MessageBus(format!("Agent {} not registered", message.to)))?;

        channel
            .send(message)
            .map_err(|e| Error::MessageBus(format!("Failed to send message: {}", e)))?;

        Ok(())
    }

    /// Broadcast a message to all registered agents
    pub async fn broadcast(&self, message: Message) -> Result<()> {
        for channel in self.channels.values() {
            channel
                .send(message.clone())
                .map_err(|e| Error::MessageBus(format!("Failed to broadcast: {}", e)))?;
        }
        Ok(())
    }

    /// Get the number of registered agents
    pub fn agent_count(&self) -> usize {
        self.channels.len()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let from = AgentId::new();
        let to = AgentId::new();
        let msg = Message::start(from, to);

        assert_eq!(msg.from, from);
        assert_eq!(msg.to, to);
        assert!(matches!(msg.payload, MessagePayload::Start));
    }

    #[test]
    fn test_message_bus_registration() {
        let mut bus = MessageBus::new();
        let agent_id = AgentId::new();

        let _rx = bus.register(agent_id);
        assert_eq!(bus.agent_count(), 1);

        bus.unregister(&agent_id);
        assert_eq!(bus.agent_count(), 0);
    }

    #[tokio::test]
    async fn test_message_bus_send() {
        let mut bus = MessageBus::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let mut rx = bus.register(agent2);
        let _tx = bus.register(agent1);

        let msg = Message::start(agent1, agent2);
        bus.send(msg.clone()).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, msg.id);
        assert_eq!(received.from, agent1);
    }
}
