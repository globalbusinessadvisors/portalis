"""
Interactive Translation API Model for Triton
Provides real-time, streaming translation with context awareness
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import List, Dict, Any, Optional
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    Interactive API Model for real-time translation
    Supports streaming, context preservation, and multi-turn interactions
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """Initialize interactive translation service"""
        logger.info("Initializing Interactive API Model...")

        self.model_config = json.loads(args['model_config'])

        # Get configuration
        self.enable_streaming = self._get_parameter('ENABLE_STREAMING', 'true') == 'true'
        self.max_context_length = int(self._get_parameter('MAX_CONTEXT_LENGTH', '8192'))

        # Initialize session management
        self.active_sessions = {}  # session_id -> context
        self.session_cache = {}    # session_id -> cached results

        # Context window management
        self.context_window_size = 5  # Keep last 5 interactions

        # Initialize translation engine (shared with translation_model)
        self._init_translation_engine()

        # Performance tracking
        self.metrics = {
            'active_sessions': 0,
            'total_interactions': 0,
            'avg_latency_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("Interactive API Model initialized")

    def _get_parameter(self, key: str, default: str) -> str:
        """Extract parameter from model config"""
        params = self.model_config.get('parameters', {})
        for param in params:
            if param.get('key') == key:
                return param.get('value', {}).get('string_value', default)
        return default

    def _init_translation_engine(self) -> None:
        """Initialize translation engine with caching"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Interactive API using device: {self.device}")

            # For now, use a placeholder
            # In production, load optimized model for interactive use
            self.translation_engine = None
            logger.warning("Translation engine placeholder - integrate with NeMo service")

        except Exception as e:
            logger.error(f"Error initializing translation engine: {e}")
            self.translation_engine = None

    def execute(self, requests: List) -> List:
        """
        Execute interactive translation requests

        Args:
            requests: List of inference requests with sequence control

        Returns:
            List of inference responses
        """
        responses = []

        for request in requests:
            try:
                # Extract sequence controls
                sequence_id = self._get_sequence_id(request)
                is_start = self._is_sequence_start(request)
                is_end = self._is_sequence_end(request)

                # Handle session lifecycle
                if is_start:
                    self._start_session(sequence_id)

                # Process request
                result = self._process_interactive_request(request, sequence_id)

                # Create response
                response = self._create_response(result)
                responses.append(response)

                # Cleanup if session ended
                if is_end:
                    self._end_session(sequence_id)

            except Exception as e:
                logger.error(f"Error processing interactive request: {e}")
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)

        return responses

    def _get_sequence_id(self, request) -> int:
        """Extract sequence ID from request"""
        try:
            correlation_id = request.correlation_id()
            return correlation_id if correlation_id else 0
        except:
            return 0

    def _is_sequence_start(self, request) -> bool:
        """Check if this is a sequence start"""
        try:
            start_tensor = pb_utils.get_input_tensor_by_name(request, "START")
            if start_tensor is not None:
                return bool(start_tensor.as_numpy()[0])
        except:
            pass
        return False

    def _is_sequence_end(self, request) -> bool:
        """Check if this is a sequence end"""
        try:
            end_tensor = pb_utils.get_input_tensor_by_name(request, "END")
            if end_tensor is not None:
                return bool(end_tensor.as_numpy()[0])
        except:
            pass
        return False

    def _start_session(self, session_id: int) -> None:
        """Initialize a new interaction session"""
        logger.info(f"Starting session {session_id}")
        self.active_sessions[session_id] = {
            'context': deque(maxlen=self.context_window_size),
            'start_time': time.time(),
            'interaction_count': 0
        }
        self.session_cache[session_id] = {}
        self.metrics['active_sessions'] += 1

    def _end_session(self, session_id: int) -> None:
        """Cleanup session resources"""
        logger.info(f"Ending session {session_id}")
        if session_id in self.active_sessions:
            session_duration = time.time() - self.active_sessions[session_id]['start_time']
            logger.info(
                f"Session {session_id} duration: {session_duration:.2f}s, "
                f"interactions: {self.active_sessions[session_id]['interaction_count']}"
            )
            del self.active_sessions[session_id]
            del self.session_cache[session_id]
            self.metrics['active_sessions'] -= 1

    def _process_interactive_request(
        self,
        request,
        session_id: int
    ) -> Dict[str, Any]:
        """Process a single interactive translation request"""
        start_time = time.time()

        # Extract inputs
        code_snippet_tensor = pb_utils.get_input_tensor_by_name(request, "code_snippet")
        code_snippet = code_snippet_tensor.as_numpy()[0].decode('utf-8')

        target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "target_language")
        target_language = target_lang_tensor.as_numpy()[0].decode('utf-8')

        # Extract optional context
        context = []
        try:
            context_tensor = pb_utils.get_input_tensor_by_name(request, "context")
            if context_tensor is not None:
                context_data = context_tensor.as_numpy()
                context = [c.decode('utf-8') for c in context_data]
        except:
            pass

        # Check cache
        cache_key = self._compute_cache_key(code_snippet, target_language, context)
        cached_result = self._check_cache(session_id, cache_key)
        if cached_result:
            self.metrics['cache_hits'] += 1
            logger.info(f"Cache hit for session {session_id}")
            return cached_result

        self.metrics['cache_misses'] += 1

        # Get session context
        session_context = self._get_session_context(session_id)

        # Perform translation
        result = self._translate_interactive(
            code_snippet,
            target_language,
            context + session_context
        )

        # Update session context
        self._update_session_context(session_id, code_snippet, result['translated_code'])

        # Cache result
        self._cache_result(session_id, cache_key, result)

        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(processing_time)

        result['processing_time_ms'] = processing_time * 1000
        return result

    def _compute_cache_key(
        self,
        code: str,
        target: str,
        context: List[str]
    ) -> str:
        """Compute cache key for request"""
        import hashlib
        combined = f"{code}|{target}|{''.join(context)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _check_cache(self, session_id: int, cache_key: str) -> Optional[Dict]:
        """Check session cache for result"""
        if session_id in self.session_cache:
            return self.session_cache[session_id].get(cache_key)
        return None

    def _cache_result(
        self,
        session_id: int,
        cache_key: str,
        result: Dict[str, Any]
    ) -> None:
        """Cache translation result"""
        if session_id in self.session_cache:
            # Limit cache size per session
            if len(self.session_cache[session_id]) > 100:
                # Remove oldest entry
                oldest_key = next(iter(self.session_cache[session_id]))
                del self.session_cache[session_id][oldest_key]

            self.session_cache[session_id][cache_key] = result

    def _get_session_context(self, session_id: int) -> List[str]:
        """Get accumulated context for session"""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]['context']
            return list(context)
        return []

    def _update_session_context(
        self,
        session_id: int,
        input_code: str,
        output_code: str
    ) -> None:
        """Update session context with new interaction"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['context'].append(f"Input: {input_code[:100]}...")
            session['context'].append(f"Output: {output_code[:100]}...")
            session['interaction_count'] += 1

    def _translate_interactive(
        self,
        code_snippet: str,
        target_language: str,
        context: List[str]
    ) -> Dict[str, Any]:
        """
        Perform interactive translation with context awareness

        Args:
            code_snippet: Code to translate
            target_language: Target language (e.g., 'rust')
            context: Previous interaction context

        Returns:
            Translation result with suggestions and warnings
        """
        # Build context-aware prompt
        context_str = '\n'.join(context[-3:]) if context else ''  # Last 3 contexts
        full_input = f"{context_str}\n\n{code_snippet}" if context_str else code_snippet

        # Perform translation (placeholder for NeMo integration)
        if target_language.lower() == 'rust':
            translated_code = self._quick_translate_to_rust(code_snippet)
        else:
            translated_code = code_snippet  # Passthrough for unsupported languages

        # Generate suggestions
        suggestions = self._generate_suggestions(code_snippet, translated_code)

        # Detect warnings
        warnings = self._detect_warnings(code_snippet, translated_code)

        # Calculate confidence
        confidence = self._calculate_confidence(code_snippet, translated_code, context)

        return {
            'translated_code': translated_code,
            'confidence': confidence,
            'suggestions': suggestions,
            'warnings': warnings
        }

    def _quick_translate_to_rust(self, python_code: str) -> str:
        """Quick translation for interactive use (placeholder)"""
        # Simple heuristic translation
        rust_code = python_code
        rust_code = rust_code.replace('def ', 'fn ')
        rust_code = rust_code.replace(':', ' {')
        rust_code = rust_code.replace('self', '&self')
        return rust_code + '\n}'

    def _generate_suggestions(self, input_code: str, output_code: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Check for common patterns
        if 'error' in output_code.lower() or 'result' in output_code.lower():
            suggestions.append("Consider using Result<T, E> for error handling")

        if 'async' in input_code:
            suggestions.append("Use async/await with tokio for async operations")

        if 'list' in input_code.lower() or '[' in input_code:
            suggestions.append("Consider using Vec<T> for dynamic arrays")

        return suggestions

    def _detect_warnings(self, input_code: str, output_code: str) -> List[str]:
        """Detect potential issues in translation"""
        warnings = []

        # Check for unsupported features
        if 'eval(' in input_code or 'exec(' in input_code:
            warnings.append("Dynamic code execution cannot be translated")

        if 'globals()' in input_code or 'locals()' in input_code:
            warnings.append("Reflection not directly supported in Rust")

        if input_code.count('\n') > 50:
            warnings.append("Large code snippet - consider breaking into smaller functions")

        return warnings

    def _calculate_confidence(
        self,
        input_code: str,
        output_code: str,
        context: List[str]
    ) -> float:
        """Calculate translation confidence score"""
        confidence = 0.7  # Base confidence

        # Boost confidence with context
        if context:
            confidence += min(0.15, len(context) * 0.03)

        # Reduce confidence for complex code
        if input_code.count('\n') > 30:
            confidence -= 0.1

        # Reduce confidence if warnings present
        warnings = self._detect_warnings(input_code, output_code)
        confidence -= len(warnings) * 0.05

        return max(0.1, min(1.0, confidence))

    def _create_response(self, result: Dict[str, Any]) -> Any:
        """Create Triton inference response"""
        # Create output tensors
        translated_code_tensor = pb_utils.Tensor(
            "translated_code",
            np.array([result['translated_code'].encode('utf-8')], dtype=object)
        )

        confidence_tensor = pb_utils.Tensor(
            "confidence",
            np.array([result['confidence']], dtype=np.float32)
        )

        suggestions_array = np.array(
            [s.encode('utf-8') for s in result['suggestions']],
            dtype=object
        )
        suggestions_tensor = pb_utils.Tensor("suggestions", suggestions_array)

        warnings_array = np.array(
            [w.encode('utf-8') for w in result['warnings']],
            dtype=object
        )
        warnings_tensor = pb_utils.Tensor("warnings", warnings_array)

        return pb_utils.InferenceResponse(
            output_tensors=[
                translated_code_tensor,
                confidence_tensor,
                suggestions_tensor,
                warnings_tensor
            ]
        )

    def _update_metrics(self, processing_time: float) -> None:
        """Update performance metrics"""
        self.metrics['total_interactions'] += 1
        total = self.metrics['total_interactions']
        old_avg = self.metrics['avg_latency_ms']
        self.metrics['avg_latency_ms'] = (
            (old_avg * (total - 1) + processing_time * 1000) / total
        )

    def finalize(self) -> None:
        """Cleanup on model unload"""
        logger.info("Finalizing Interactive API Model...")
        logger.info(f"Final metrics: {json.dumps(self.metrics, indent=2)}")

        # Clear all active sessions
        for session_id in list(self.active_sessions.keys()):
            self._end_session(session_id)

        logger.info("Interactive API Model finalized")
