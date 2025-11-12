"""
Advanced Context Management for Extended Conversations
Handles long context windows, memory compression, and conversation history
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float
    embedding: Optional[xp.ndarray] = None
    importance_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'importance_score': self.importance_score
        }

class ContextCompressor:
    """Compresses long contexts while preserving important information"""
    
    def __init__(self, compression_ratio: float = 0.7):
        self.compression_ratio = compression_ratio
        
    def compress_text(self, text: str, importance_scores: Optional[xp.ndarray] = None) -> str:
        """Compress text by removing less important parts"""
        sentences = text.split('. ')
        
        if importance_scores is None:
            # Simple heuristic: keep first and last sentences, and some middle ones
            if len(sentences) <= 3:
                return text
            
            # Keep first, last, and every other sentence
            compressed = [sentences[0]]
            for i in range(1, len(sentences)-1, 2):
                compressed.append(sentences[i])
            compressed.append(sentences[-1])
            
            return '. '.join(compressed)
        else:
            # Use importance scores to select sentences
            if len(importance_scores) != len(sentences):
                return text
                
            # Sort sentences by importance and keep top ones
            sentence_importance = list(zip(sentences, importance_scores))
            sentence_importance.sort(key=lambda x: x[1], reverse=True)
            
            keep_count = max(1, int(len(sentences) * self.compression_ratio))
            kept_sentences = [s[0] for s in sentence_importance[:keep_count]]
            
            # Restore original order
            kept_sentences.sort(key=lambda x: sentences.index(x))
            
            return '. '.join(kept_sentences)
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        # Simple keyword-based extraction
        key_phrases = [
            "important", "key", "main", "crucial", "essential",
            "remember", "note", "summary", "conclusion",
            "agree", "disagree", "think", "believe", "understand"
        ]
        
        sentences = text.split('. ')
        key_points = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(phrase in sentence_lower for phrase in key_phrases):
                key_points.append(sentence.strip())
        
        return key_points[:5]  # Limit to top 5 key points

class SlidingWindowManager:
    """Manages sliding context windows for long conversations"""
    
    def __init__(self, max_tokens: int = 4096, overlap_tokens: int = 512):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
    def create_windows(self, token_ids: List[int]) -> List[List[int]]:
        """Create overlapping windows from token sequence"""
        if len(token_ids) <= self.max_tokens:
            return [token_ids]
        
        windows = []
        step_size = self.max_tokens - self.overlap_tokens
        
        for i in range(0, len(token_ids), step_size):
            window = token_ids[i:i + self.max_tokens]
            if len(window) > self.overlap_tokens:  # Only keep substantial windows
                windows.append(window)
            
            if i + self.max_tokens >= len(token_ids):
                break
        
        return windows
    
    def merge_window_outputs(self, window_outputs: List[xp.ndarray], 
                           merge_strategy: str = "average") -> xp.ndarray:
        """Merge outputs from multiple windows"""
        if not window_outputs:
            return np.array([])
        
        if merge_strategy == "average":
            return np.mean(window_outputs, axis=0)
        elif merge_strategy == "weighted_average":
            # Give more weight to middle windows
            weights = self._get_window_weights(len(window_outputs))
            weighted_sum = np.zeros_like(window_outputs[0])
            for i, output in enumerate(window_outputs):
                weighted_sum += weights[i] * output
            return weighted_sum
        else:
            return window_outputs[-1]  # Use last window
    
    def _get_window_weights(self, num_windows: int) -> List[float]:
        """Generate weights for window merging"""
        if num_windows == 1:
            return [1.0]
        
        # Gaussian-like weights centered in the middle
        center = (num_windows - 1) / 2
        weights = []
        
        for i in range(num_windows):
            distance = abs(i - center)
            weight = np.exp(-0.5 * (distance / center) ** 2)
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        return [w / total for w in weights]

class ConversationMemory:
    """Advanced conversation memory management"""
    
    def __init__(self, max_turns: int = 100, max_tokens: int = 8192):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.conversation_history: List[ConversationTurn] = []
        self.compressor = ContextCompressor()
        self.window_manager = SlidingWindowManager(max_tokens)
        
    def add_turn(self, role: str, content: str, embedding: Optional[xp.ndarray] = None,
                importance_score: float = 0.5) -> None:
        """Add a new conversation turn"""
        import time
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            embedding=embedding,
            importance_score=importance_score
        )
        self.conversation_history.append(turn)
        
        # Maintain limits
        self._enforce_limits()
    
    def get_recent_context(self, num_turns: int = 10, include_system: bool = True) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-num_turns:]
        context_parts = []
        
        for turn in recent_turns:
            if not include_system and turn.role == "system":
                continue
            context_parts.append(f"{turn.role.capitalize()}: {turn.content}")
        
        return "\n".join(context_parts)
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Get context relevant to the current query"""
        if not self.conversation_history:
            return ""
        
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        scored_turns = []
        
        for turn in self.conversation_history:
            if turn.role == "system":
                continue
                
            turn_words = set(turn.content.lower().split())
            overlap = len(query_words.intersection(turn_words))
            relevance = overlap / max(len(query_words), 1)
            
            # Boost recent turns
            time_decay = 1.0 / (1.0 + (time.time() - turn.timestamp) / 3600)  # 1 hour half-life
            final_score = relevance * time_decay * turn.importance_score
            
            scored_turns.append((turn, final_score))
        
        # Sort by relevance and get top turns
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        relevant_turns = [turn for turn, _ in scored_turns[:top_k]]
        
        # Sort by time to maintain chronological order
        relevant_turns.sort(key=lambda x: x.timestamp)
        
        context_parts = []
        for turn in relevant_turns:
            context_parts.append(f"{turn.role.capitalize()}: {turn.content}")
        
        return "\n".join(context_parts)
    
    def get_compressed_summary(self, max_length: int = 500) -> str:
        """Get a compressed summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history."
        
        # Combine all content
        all_content = " ".join([turn.content for turn in self.conversation_history])
        
        if len(all_content) <= max_length:
            return all_content
        
        # Compress the content
        compressed = self.compressor.compress_text(all_content)
        
        # Extract key points
        key_points = self.compressor.extract_key_points(all_content)
        
        summary = f"Conversation summary: {compressed[:max_length]}"
        if key_points:
            summary += f"\nKey points: {'; '.join(key_points)}"
        
        return summary
    
    def _enforce_limits(self) -> None:
        """Enforce memory limits"""
        # Limit number of turns
        if len(self.conversation_history) > self.max_turns:
            # Remove oldest turns, but keep system messages
            non_system_turns = [t for t in self.conversation_history if t.role != "system"]
            system_turns = [t for t in self.conversation_history if t.role == "system"]
            
            if len(non_system_turns) > self.max_turns - len(system_turns):
                remove_count = len(non_system_turns) - (self.max_turns - len(system_turns))
                non_system_turns = non_system_turns[remove_count:]
            
            self.conversation_history = system_turns + non_system_turns
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.conversation_history.clear()
    
    def export_memory(self) -> Dict[str, Any]:
        """Export conversation memory for persistence"""
        return {
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history]
        }
    
    def import_memory(self, data: Dict[str, Any]) -> None:
        """Import conversation memory"""
        self.max_turns = data.get("max_turns", self.max_turns)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        
        # Reconstruct conversation history
        self.conversation_history = []
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn(
                role=turn_data["role"],
                content=turn_data["content"],
                timestamp=turn_data["timestamp"],
                importance_score=turn_data.get("importance_score", 0.5)
            )
            self.conversation_history.append(turn)

class EnhancedContextManager:
    """Main context manager that combines all components"""
    
    def __init__(self, max_context_tokens: int = 8192, max_conversation_turns: int = 100):
        self.max_context_tokens = max_context_tokens
        self.memory = ConversationMemory(max_conversation_turns, max_context_tokens)
        self.compressor = ContextCompressor()
        self.window_manager = SlidingWindowManager(max_context_tokens)
        
    def process_input(self, user_input: str, system_context: Optional[str] = None) -> Dict[str, Any]:
        """Process user input with full context management"""
        
        # Add system context if provided
        if system_context:
            self.memory.add_turn("system", system_context, importance_score=1.0)
        
        # Get relevant context for this input
        relevant_context = self.memory.get_relevant_context(user_input)
        recent_context = self.memory.get_recent_context(num_turns=5)
        
        # Combine contexts intelligently
        if relevant_context and recent_context:
            # Avoid duplication
            relevant_lines = set(relevant_context.split('\n'))
            recent_lines = set(recent_context.split('\n'))
            unique_recent = [line for line in recent_lines if line not in relevant_lines]
            
            full_context = relevant_context
            if unique_recent:
                full_context += f"\n\nRecent context:\n" + "\n".join(unique_recent)
        else:
            full_context = relevant_context or recent_context
        
        # Add user input to memory
        self.memory.add_turn("user", user_input, importance_score=0.8)
        
        return {
            "full_context": full_context,
            "relevant_context": relevant_context,
            "recent_context": recent_context,
            "conversation_summary": self.memory.get_compressed_summary(),
            "memory_stats": {
                "total_turns": len(self.memory.conversation_history),
                "recent_turns": min(5, len(self.memory.conversation_history))
            }
        }
    
    def add_response(self, response: str, embedding: Optional[xp.ndarray] = None) -> None:
        """Add AI response to memory"""
        self.memory.add_turn("assistant", response, embedding=embedding, importance_score=0.7)
    
    def get_context_for_model(self, max_tokens: int = 4096) -> str:
        """Get context formatted for the transformer model"""
        # Get recent and relevant context
        recent = self.memory.get_recent_context(num_turns=10)
        summary = self.memory.get_compressed_summary(max_length=200)
        
        # Combine with token limit consideration
        context = f"Conversation Summary: {summary}\n\nRecent Context:\n{recent}"
        
        # Simple token estimation (rough approximation)
        estimated_tokens = len(context.split()) * 1.3
        if estimated_tokens > max_tokens * 0.8:
            # Compress if too long
            context = self.compressor.compress_text(context)
        
        return context
    
    def reset_conversation(self) -> None:
        """Reset the conversation memory"""
        self.memory.clear_memory()
    
    def export_conversation(self) -> str:
        """Export conversation for persistence"""
        return json.dumps(self.memory.export_memory(), indent=2)
    
    def import_conversation(self, conversation_data: str) -> None:
        """Import conversation from persistence"""
        try:
            data = json.loads(conversation_data)
            self.memory.import_memory(data)
        except json.JSONDecodeError:
            print("❌ Invalid conversation data format")