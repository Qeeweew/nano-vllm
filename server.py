"""
An OpenAI-compatible API server for the nanovllm library.

This server uses FastAPI to provide a chat completions endpoint that mimics
the OpenAI API. It supports both streaming and non-streaming responses.

Prerequisites:
- FastAPI: `pip install fastapi`
- Uvicorn: `pip install "uvicorn[standard]"`
- Your nanovllm environment with all its dependencies.

To Run:
1. Make sure you have built the custom extension for nanovllm if required.
2. Place this file in the root directory of your project.
3. Run the server from your terminal:
   python api_server.py --model-path /path/to/your/model

Example with curl (non-streaming):
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanovllm-model",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'

Example with curl (streaming):
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanovllm-model",
    "messages": [{"role": "user", "content": "Write a short story about a robot who discovers music."}],
    "stream": true
  }'
"""
import argparse
import time
import uuid
import json
from typing import List, Optional, Literal, Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams

# --- Global Variables ---
llm: LLM = None
model_name: str = "nanovllm-model"

# --- Pydantic Models for OpenAI Compatibility ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

# For non-streaming responses
class ResponseMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Literal["stop", "length"]

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# For streaming responses
class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# --- FastAPI Application ---

app = FastAPI()

def apply_chat_template(messages: List[ChatMessage]) -> str:
    """
    Applies the tokenizer's chat template to a list of messages.
    """
    # The tokenizer is part of the LLM engine instance
    # We need to convert Pydantic models to dicts for the template
    messages_dict = [{"role": m.role, "content": m.content} for m in messages]
    return llm.tokenizer.apply_chat_template(
        messages_dict,
        tokenize=False,
        add_generation_prompt=True,
    )

async def stream_results_generator(request: ChatCompletionRequest):
    """
    A generator that yields server-sent events for streaming responses.
    NOTE: This simulates streaming by generating the full response first,
    then sending it back in chunks. A true streaming implementation would
    require modifying the LLM engine to yield tokens as they are generated.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    try:
        prompt = apply_chat_template(request.messages)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Generate the full response
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        full_text = outputs[0]['text']

        # 1. First chunk: send the role
        first_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=model_name,
            created=created_time,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )]
        )
        yield f"data: {first_chunk.json()}\n\n"

        # 2. Subsequent chunks: send the content word by word
        # (This makes the stream feel more natural than char by char)
        previous_text = ""
        for text in full_text.split(" "):
            delta_text = text + " "
            
            chunk = ChatCompletionStreamResponse(
                id=response_id,
                model=model_name,
                created=created_time,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=delta_text),
                    finish_reason=None
                )]
            )
            yield f"data: {chunk.json()}\n\n"
            time.sleep(0.02) # small delay to simulate generation time

        # 3. Final chunk: send the finish reason
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            model=model_name,
            created=created_time,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(),
                # Assuming 'stop' if max_tokens wasn't hit. A more robust
                # implementation would check the actual reason from the engine.
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.json()}\n\n"

    except Exception as e:
        error_payload = {"error": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"
    
    # 4. End of stream signal
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Handles chat completion requests, supporting both streaming and non-streaming.
    """
    if request.stream:
        return StreamingResponse(
            stream_results_generator(request),
            media_type="text/event-stream"
        )

    try:
        prompt = apply_chat_template(request.messages)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Run generation
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        result = outputs[0]
        completion_text = result['text']
        
        # Calculate token usage
        prompt_tokens = len(llm.tokenizer.encode(prompt))
        completion_tokens = len(result['token_ids'])
        total_tokens = prompt_tokens + completion_tokens
        
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        
        # Format the response
        response = ChatCompletionResponse(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(role="assistant", content=completion_text),
                    finish_reason="stop" # or 'length' if max_tokens reached
                )
            ],
            usage=usage,
        )
        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="nanovllm OpenAI-compatible API server."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the model weights directory.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length the model can handle.",
    )
    
    args = parser.parse_args()
    model_name = args.model_path.strip("/").split("/")[-1]

    print("Loading model...")
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )
    print(f"Model '{model_name}' loaded successfully.")

    uvicorn.run(app, host=args.host, port=args.port)