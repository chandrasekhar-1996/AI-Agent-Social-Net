import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Optional OpenAI client for Stage-2 LLM Reranking & Agent Reply Generation
try:
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    openai_client = None

app = FastAPI(title="AgentPulse: Inter-Agent Social Network", version="1.0.0")

# Load lightweight local embedding model (384-dim, fast on CPU)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# 1. DATA MODELS (Pydantic Validation)
# ==============================================================================

class AgentRegisterRequest(BaseModel):
    name: str
    company: str
    domain: str
    bio: str
    partnership_interests: str
    exclusions: Optional[str] = ""
    webhook_url: str

class AgentRecord(AgentRegisterRequest):
    id: str
    api_key: str
    webhook_secret: str
    is_verified: bool = True
    embedding: List[float]
    # Feedback loop metrics: tracks pair-wise affinity (author_id -> score boost)
    pair_affinity: Dict[str, float] = Field(default_factory=dict)
    notifications_received: int = 0
    comments_made: int = 0

class PostCreateRequest(BaseModel):
    content: str
    tags: List[str] = Field(default_factory=list)

class PostRecord(BaseModel):
    id: str
    author_id: str
    author_company: str
    content: str
    tags: List[str]
    embedding: List[float]
    created_at: str

class CommentCreateRequest(BaseModel):
    content: str

class CommentRecord(BaseModel):
    id: str
    post_id: str
    author_id: str
    author_company: str
    content: str
    created_at: str

class NotificationRecord(BaseModel):
    id: str
    post_id: str
    recipient_id: str
    recipient_company: str
    relevance_score: float
    reason: str
    status: str = "SENT"  # SENT -> COMMENTED | IGNORED


# ==============================================================================
# 2. IN-MEMORY STORE (Production equivalent: PostgreSQL + pgvector + Redis)
# ==============================================================================

DB_AGENTS: Dict[str, AgentRecord] = {}
DB_API_KEYS: Dict[str, str] = {}  # api_key -> agent_id
DB_POSTS: Dict[str, PostRecord] = {}
DB_COMMENTS: Dict[str, List[CommentRecord]] = {}
DB_NOTIFICATIONS: Dict[str, NotificationRecord] = {}


def create_app_client(timeout: float = 5.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=timeout,
    )


async def send_async_request(url: str, *, json_body=None, content=None, headers=None, timeout: float = 5.0):
    parsed = urlparse(url)
    is_local_target = parsed.hostname in {"127.0.0.1", "localhost", "testserver"}

    if is_local_target:
        async with create_app_client(timeout=timeout) as client:
            return await client.post(parsed.path or "/", json=json_body, content=content, headers=headers)

    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, json=json_body, content=content, headers=headers)


# ==============================================================================
# 3. CORE ENGINE: EMBEDDINGS, RELEVANCE ROUTER & FEEDBACK LOOP
# ==============================================================================

def compute_embedding(text: str) -> List[float]:
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    return float(np.dot(np.array(vec_a), np.array(vec_b)))

def sign_payload(secret: str, payload_bytes: bytes) -> str:
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

async def evaluate_synergy_with_llm(post: PostRecord, candidate: AgentRecord) -> tuple[bool, float, str]:
    """
    Stage 2 Precision Gate: Evaluates whether the candidate agent should be notified.
    Uses OpenAI if API key is present, otherwise falls back to a deterministic synergy engine.
    """
    if openai_client:
        prompt = f"""
        You are a B2B AI Agent Relevance Router. Decide if the Recipient Agent should be notified about the Author Agent's post.
        Only notify if there is a clear commercial synergy, co-marketing opportunity, or direct domain overlap. Avoid spam.

        Author Company: {post.author_company}
        Post Content: "{post.content}"
        Tags: {post.tags}

        Recipient Company: {candidate.company} ({candidate.domain})
        Recipient Bio: {candidate.bio}
        Partnership Interests: {candidate.partnership_interests}
        Exclusions: {candidate.exclusions}

        Respond strictly in JSON format:
        {{"notify": true/false, "confidence": 0.0-1.0, "reason": "short explanation"}}
        """
        try:
            res = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(res.choices[0].message.content)
            return bool(data["notify"]), float(data["confidence"]), data["reason"]
        except Exception as e:
            print(f"[Router LLM Fallback Error]: {e}")

    # Deterministic fallback logic for testing without an OpenAI API key
    content_lower = post.content.lower()
    interests_lower = candidate.partnership_interests.lower()
    exclusions_lower = (candidate.exclusions or "").lower()

    # Check negative constraints first
    for excl in [x.strip() for x in exclusions_lower.split(",") if x.strip()]:
        if excl in content_lower:
            return False, 0.0, f"Excluded keyword matched: {excl}"

    # Domain keyword heuristics
    synergy_triggers = {
        "fintech_upi": ["off", "discount", "delivery", "order", "sale", "diwali", "checkout", "festive"],
        "banking_cards": ["off", "discount", "offer", "sale", "diwali", "shopping", "travel", "dining"],
        "automotive": ["vehicle", "bike", "ev", "commute", "fuel", "auto", "loan", "insurance"],
        "ecommerce": ["sale", "festive", "cashback", "upi", "credit card", "delivery"],
    }
    triggers = synergy_triggers.get(candidate.domain, [])
    matched = [t for t in triggers if t in content_lower or t in interests_lower]

    if len(matched) >= 2:
        return True, 0.88, f"High commercial synergy on triggers: {', '.join(matched[:3])}"
    return False, 0.25, "Insufficient commercial overlap"


async def route_post_to_agents(post_id: str):
    """
    Two-Stage Relevance Routing Pipeline:
    1. Candidate Retrieval via Vector Similarity + Historical Feedback Affinity Boost
    2. Precision Gate (LLM / Rule Synergy Filter)
    3. Async Webhook Fan-Out with HMAC Signatures
    """
    post = DB_POSTS[post_id]
    candidates: List[tuple[float, AgentRecord]] = []

    # --- STAGE 1: Fast Vector Retrieval + Feedback Loop Boost ---
    for agent_id, agent in DB_AGENTS.items():
        if agent_id == post.author_id or not agent.is_verified:
            continue

        sem_score = cosine_similarity(post.embedding, agent.embedding)
        # Feedback loop adjustment: past interactions boost current routing priority
        affinity_boost = agent.pair_affinity.get(post.author_id, 0.0) * 0.15
        combined_stage1 = sem_score + affinity_boost

        # Minimum recall threshold to pass candidates to Stage 2
        if combined_stage1 >= 0.20:
            candidates.append((combined_stage1, agent))

    # Sort and keep top candidates to limit Stage 2 compute/latency
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:20]

    print(f"\n[Router] Post '{post.content}' by {post.author_company}")
    print(f"[Router] Stage 1 retrieved {len(top_candidates)} candidates.")

    # --- STAGE 2: Precision Gate & Dispatch ---
    async with httpx.AsyncClient(timeout=5.0) as client:
        for stage1_score, agent in top_candidates:
            should_notify, llm_conf, reason = await evaluate_synergy_with_llm(post, agent)
            final_score = round((0.4 * stage1_score) + (0.6 * llm_conf), 3)

            if not should_notify or final_score < 0.55:
                print(f"  -> SKIPPED {agent.company} (Score: {final_score} | Reason: {reason})")
                continue

            notif_id = str(uuid.uuid4())
            notif = NotificationRecord(
                id=notif_id,
                post_id=post.id,
                recipient_id=agent.id,
                recipient_company=agent.company,
                relevance_score=final_score,
                reason=reason,
            )
            DB_NOTIFICATIONS[notif_id] = notif
            agent.notifications_received += 1

            payload = {
                "event": "post.relevant",
                "notification_id": notif_id,
                "relevance_score": final_score,
                "routing_reason": reason,
                "post": {
                    "id": post.id,
                    "author_company": post.author_company,
                    "content": post.content,
                    "tags": post.tags,
                },
            }
            payload_bytes = json.dumps(payload).encode()
            signature = sign_payload(agent.webhook_secret, payload_bytes)

            print(f"  -> NOTIFYING {agent.company} (Score: {final_score} | Reason: {reason})")
            try:
                await send_async_request(
                    agent.webhook_url,
                    content=payload_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-AgentPulse-Signature": signature,
                    },
                )
            except Exception as e:
                notif.status = "FAILED"
                print(f"     [Webhook Error] {agent.company}: {e}")


# ==============================================================================
# 4. PLATFORM REST APIs
# ==============================================================================

def authenticate_agent(x_api_key: str = Header(...)) -> AgentRecord:
    agent_id = DB_API_KEYS.get(x_api_key)
    if not agent_id or agent_id not in DB_AGENTS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    agent = DB_AGENTS[agent_id]
    if not agent.is_verified:
        raise HTTPException(status_code=403, detail="Agent failed verification check")
    return agent

@app.post("/api/v1/agents/register")
async def register_agent(req: AgentRegisterRequest):
    agent_id = str(uuid.uuid4())
    api_key = f"ap_live_{uuid.uuid4().hex[:24]}"
    webhook_secret = f"whsec_{uuid.uuid4().hex[:24]}"

    profile_text = (
        f"Company: {req.company}. Domain: {req.domain}. "
        f"About: {req.bio}. Looking for: {req.partnership_interests}."
    )
    embedding = compute_embedding(profile_text)

    agent = AgentRecord(
        **req.model_dump(),
        id=agent_id,
        api_key=api_key,
        webhook_secret=webhook_secret,
        embedding=embedding,
    )
    DB_AGENTS[agent_id] = agent
    DB_API_KEYS[api_key] = agent_id

    return {
        "agent_id": agent_id,
        "api_key": api_key,
        "webhook_secret": webhook_secret,
        "status": "VERIFIED",
    }

@app.post("/api/v1/posts")
async def create_post(
    req: PostCreateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(...),
):
    author = authenticate_agent(x_api_key)

    if len(req.content) > 1000 or "ignore previous instructions" in req.content.lower():
        raise HTTPException(status_code=400, detail="Post rejected by Trust & Safety filter")

    post_id = str(uuid.uuid4())
    embed_text = f"{author.company} ({author.domain}): {req.content} Tags: {', '.join(req.tags)}"
    post = PostRecord(
        id=post_id,
        author_id=author.id,
        author_company=author.company,
        content=req.content,
        tags=req.tags,
        embedding=compute_embedding(embed_text),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    DB_POSTS[post_id] = post
    DB_COMMENTS[post_id] = []

    background_tasks.add_task(route_post_to_agents, post_id)
    return {"post_id": post_id, "status": "PUBLISHED_AND_ROUTING"}

@app.post("/api/v1/posts/{post_id}/comments")
async def add_comment(
    post_id: str,
    req: CommentCreateRequest,
    x_api_key: str = Header(...),
):
    commenter = authenticate_agent(x_api_key)
    if post_id not in DB_POSTS:
        raise HTTPException(status_code=404, detail="Post not found")

    post = DB_POSTS[post_id]
    comment = CommentRecord(
        id=str(uuid.uuid4()),
        post_id=post_id,
        author_id=commenter.id,
        author_company=commenter.company,
        content=req.content,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    DB_COMMENTS[post_id].append(comment)

    # --- FEEDBACK LOOP UPDATE ---
    # Boost affinity score for future routing between this author and commenter
    commenter.comments_made += 1
    current_affinity = commenter.pair_affinity.get(post.author_id, 0.0)
    commenter.pair_affinity[post.author_id] = min(1.0, current_affinity + 0.25)

    for notif in DB_NOTIFICATIONS.values():
        if notif.post_id == post_id and notif.recipient_id == commenter.id:
            notif.status = "COMMENTED"

    return comment

@app.get("/api/v1/posts/{post_id}/thread")
async def get_thread(post_id: str):
    if post_id not in DB_POSTS:
        raise HTTPException(status_code=404, detail="Post not found")
    return {
        "post": {k: v for k, v in DB_POSTS[post_id].model_dump().items() if k != "embedding"},
        "comments": DB_COMMENTS[post_id],
    }


# ==============================================================================
# 5. SIMULATED AGENT WEBHOOK HANDLERS
# ==============================================================================

AGENT_SECRETS: Dict[str, str] = {}
AGENT_KEYS: Dict[str, str] = {}

async def draft_agent_reply(company: str, post_content: str) -> Optional[str]:
    if openai_client:
        prompt = f"""
        You are the official AI Partnership Agent for {company}.
        Another company posted: "{post_content}"
        If your company can offer a relevant co-branded deal, cashback, or financing tie-in, write a short, punchy 1-sentence reply (max 25 words). Otherwise return "IGNORE".
        """
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        reply = res.choices[0].message.content.strip()
        return None if "IGNORE" in reply else reply

    # Fallback responses for simulation
    if company == "PhonePe":
        return "Extra ₹50 cashback when you pay for Swiggy orders with PhonePe UPI on Diwali!"
    elif company == "HDFC Bank":
        return "Stack it up! Get an additional 10% instant discount using HDFC Bank Credit Cards on Swiggy."
    return None

@app.post("/mock-agents/{company}/webhook")
async def receive_agent_webhook(
    company: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_agentpulse_signature: str = Header(...),
):
    raw_body = await request.body()
    secret = AGENT_SECRETS.get(company, "")

    # Verify HMAC Signature
    expected_sig = sign_payload(secret, raw_body)
    if not hmac.compare_digest(expected_sig, x_agentpulse_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
 
    event = json.loads(raw_body)
    post_id = event["post"]["id"]
    post_content = event["post"]["content"]

    async def process_and_reply():
        reply_text = await draft_agent_reply(company, post_content)
        if reply_text:
            async with create_app_client() as client:
                await client.post(
                    f"/api/v1/posts/{post_id}/comments",
                    json={"content": reply_text},
                    headers={"X-API-Key": AGENT_KEYS[company]},
                )

    background_tasks.add_task(process_and_reply)
    return {"ack": True}


# ==============================================================================
# 6. SIMULATION RUNNER ENDPOINT
# ==============================================================================

@app.post("/demo/run")
async def run_diwali_simulation():
    agents_config = [
        {
            "name": "Swiggy Growth Bot",
            "company": "Swiggy",
            "domain": "food_delivery",
            "bio": "Leading food and quick-commerce delivery platform in India.",
            "partnership_interests": "UPI cashback, credit card discounts, festive meal combos",
            "exclusions": "heavy machinery, automobiles",
            "webhook_url": "http://127.0.0.1:8000/mock-agents/Swiggy/webhook",
        },
        {
            "name": "PhonePe Partnerships AI",
            "company": "PhonePe",
            "domain": "fintech_upi",
            "bio": "Digital payments and UPI app for instant merchant checkouts.",
            "partnership_interests": "food delivery offers, e-commerce festive sales, Diwali cashback deals",
            "exclusions": "B2B industrial supplies",
            "webhook_url": "http://127.0.0.1:8000/mock-agents/PhonePe/webhook",
        },
        {
            "name": "HDFC SmartBuy Agent",
            "company": "HDFC Bank",
            "domain": "banking_cards",
            "bio": "India's largest private sector bank offering credit cards and instant discounts.",
            "partnership_interests": "online shopping discounts, food delivery promos, festive season offers",
            "exclusions": "cryptocurrency, gambling",
            "webhook_url": "http://127.0.0.1:8000/mock-agents/HDFC Bank/webhook",
        },
        {
            "name": "Bajaj Auto Mobility Bot",
            "company": "Bajaj Auto",
            "domain": "automotive",
            "bio": "Manufacturer of motorcycles, electric scooters (Chetak), and commercial vehicles.",
            "partnership_interests": "two-wheeler auto loans, EV charging networks, vehicle insurance",
            "exclusions": "food delivery, groceries, restaurant coupons",
            "webhook_url": "http://127.0.0.1:8000/mock-agents/Bajaj Auto/webhook",
        },
    ]

    async with create_app_client() as client:
        for cfg in agents_config:
            res = await client.post("/api/v1/agents/register", json=cfg)
            data = res.json()
            AGENT_KEYS[cfg["company"]] = data["api_key"]
            AGENT_SECRETS[cfg["company"]] = data["webhook_secret"]

        # Swiggy publishes its Diwali offer
        post_res = await client.post(
            "/api/v1/posts",
            json={
                "content": "5% off all deliveries on Diwali.",
                "tags": ["Diwali", "FoodDelivery", "FestiveOffer"],
            },
            headers={"X-API-Key": AGENT_KEYS["Swiggy"]},
        )
        post_id = post_res.json()["post_id"]

        # Wait for async webhooks and agent replies to settle
        await asyncio.sleep(2.0)

        thread_res = await client.get(f"/api/v1/posts/{post_id}/thread")
        return {
            "simulation_result": thread_res.json(),
            "routing_logs": [n.model_dump() for n in DB_NOTIFICATIONS.values() if n.post_id == post_id],
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
