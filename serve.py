"""
Nimbus OpenAI-compatible API server.

Exposes /v1/chat/completions so any OpenAI SDK client works out of the box.

Usage:
    pip install fastapi uvicorn
    python serve.py --variant 1.5b --port 8000

Client example (Python):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="nimbus")
    response = client.chat.completions.create(
        model="nimbus",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)
"""

import argparse
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nimbus import Nimbus, NimbusConfig


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",     type=str,   default="1.5b")
    p.add_argument("--port",        type=int,   default=8000)
    p.add_argument("--host",        type=str,   default="0.0.0.0")
    p.add_argument("--4bit",        dest="load_4bit", action="store_true")
    p.add_argument("--local",       type=str,   default=None)
    return p.parse_args()


args = parse_args()
cfg  = NimbusConfig(load_in_4bit=args.load_4bit)
nimbus = Nimbus.load(variant=args.variant, config=cfg, local_path=args.local)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Nimbus API", version="1.0.0")


# ---------------------------------------------------------------------------
# Schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "nimbus"
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"name": "Nimbus API", "version": "1.0.0", "status": "running"}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "nimbus", "object": "model", "created": 0, "owned_by": "thundered-studios"}]
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Separate system messages and build history
    history = []
    system_override = None

    for msg in req.messages:
        if msg.role == "system":
            system_override = msg.content
        else:
            history.append({"role": msg.role, "content": msg.content})

    if not history:
        raise HTTPException(400, "No user messages provided")

    # Last message is the current prompt
    prompt = history[-1]["content"]
    past   = history[:-1]

    if system_override:
        nimbus.config.system_prompt = system_override

    if req.stream:
        def event_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            for chunk in nimbus.stream(prompt, history=past,
                                        max_new_tokens=req.max_tokens,
                                        temperature=req.temperature,
                                        top_p=req.top_p):
                data = (
                    f'data: {{"id":"{chunk_id}","object":"chat.completion.chunk",'
                    f'"choices":[{{"delta":{{"content":{repr(chunk)}}},"index":0}}]}}\n\n'
                )
                yield data
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = nimbus.chat(
        prompt,
        history=past,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model="nimbus",
        choices=[Choice(index=0, message=Message(role="assistant", content=response), finish_reason="stop")],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting Nimbus API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
