import time
import logging

logger = logging.getLogger(__name__)

# Try to import official google-genai SDK, but handle gracefully for placeholder/mock runs
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("google-genai package not available. Gemini client will run in Mock mode only.")

class GeminiClient:
    def __init__(self, model_name: str, api_key: str = "", temperature: float = 0.0, max_tokens: int = 1024, mock_mode: bool = True):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mock_mode = mock_mode
        self.api_key = api_key

        if not self.mock_mode and SDK_AVAILABLE:
            if not self.api_key:
                raise ValueError("API key must be provided if mock_mode is set to False.")
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            self.mock_mode = True  # force mock mode if SDK is missing or requested

    def call_gemini(self, prompt: str, context: str = "") -> dict:
        """Invokes the model with the combined prompt and context, tracking usage and latency.
        
        Returns:
            A dictionary containing:
                - "response_text": The model's answer.
                - "latency_seconds": Time taken for the API call.
                - "input_tokens": Estimated token count.
                - "output_tokens": Estimated token count.
        """
        combined_prompt = f"Context:\n{context}\n\nTask:\n{prompt}" if context else prompt
        
        start_time = time.time()
        
        if self.mock_mode:
            # Simulate a realistic model latency of 0.5 to 1.5 seconds
            time.sleep(0.8)
            latency = time.time() - start_time
            
            # Simple token estimators (1 token ~= 4 chars)
            input_tokens = len(combined_prompt) // 4
            mock_response = (
                f"### Mock Gemini Response\n"
                f"This is a mock baseline response simulating model generation for the task.\n\n"
                f"**Evaluation Prompt**: \"{prompt[:100]}...\"\n"
                f"**Supplied Context Size**: {len(context)} chars.\n"
                f"**Active Model**: {self.model_name}\n"
            )
            output_tokens = len(mock_response) // 4
            
            return {
                "response_text": mock_response,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
        else:
            if not self.client:
                raise RuntimeError("GenAI client not initialized.")
                
            logger.info(f"Sending prompt to Gemini API model ({self.model_name})...")
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=combined_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                )
                latency = time.time() - start_time
                
                # Retrieve counts if available from usage_metadata
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                except Exception:
                    input_tokens = len(combined_prompt) // 4
                    output_tokens = len(response.text) // 4
                    
                return {
                    "response_text": response.text,
                    "latency_seconds": latency,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            except Exception as e:
                logger.error(f"Error communicating with Gemini API: {e}")
                raise e
