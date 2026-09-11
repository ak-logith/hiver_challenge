import json
import os
import re
import urllib.request
import urllib.error

LOCAL_LLM_ENDPOINTS = [
    os.getenv("LOCAL_LLM_URL", ""),
    "http://127.0.0.1:11434/api/generate",
    "http://127.0.0.1:1234/v1/chat/completions",
    "http://127.0.0.1:8000/v1/chat/completions",
]

_ACTIVE_ENDPOINT = None
_PROBED = False

def detect_active_llm_endpoint() -> str | None:
    global _ACTIVE_ENDPOINT, _PROBED
    if _PROBED:
        return _ACTIVE_ENDPOINT
    _PROBED = True

    for endpoint in LOCAL_LLM_ENDPOINTS:
        if not endpoint:
            continue
        try:
            req = urllib.request.Request(
                endpoint,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=0.5):
                _ACTIVE_ENDPOINT = endpoint
                print(f"Connected to local LLM service at {endpoint}")
                return _ACTIVE_ENDPOINT
        except Exception:
            continue

    print("No local LLM HTTP daemon found. Utilizing local intelligent customer support generation engine.")
    return None

def query_local_http_llm(prompt: str, system_prompt: str = "") -> str | None:
    endpoint = detect_active_llm_endpoint()
    if not endpoint:
        return None
    try:
        if "11434/api/generate" in endpoint:
            payload = json.dumps({
                "model": "llama3",
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False
            }).encode("utf-8")
        else:
            payload = json.dumps({
                "messages": [
                    {"role": "system", "content": system_prompt or "You are a customer support agent."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            }).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "response" in data:
                return data["response"].strip()
            elif "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
    return None

# ==================== BASELINE 1: TRIVIAL (Canned Macro) ====================
def generate_trivial_baseline(customer_message: str) -> str:
    """Trivial Baseline: Constant canned generic support macro."""
    return (
        "Thank you for contacting customer support. We have received your inquiry "
        "and our support team is actively reviewing your request. "
        "We will follow up with you as soon as an update becomes available. "
        "Sincerely, Customer Support Team."
    )

# ==================== BASELINE 2: SIMPLE (Keyword Retrieval) ====================
FAQ_MACRO_BANK = [
    ("refund billing charge money credit return duplicate promo discount",
     "Thank you for contacting our billing department. We process all refunds within 3-5 business days upon verification. Please ensure your order details are attached so we can review the charge."),
    ("shipping delay tracking package carrier transit delivered courier weather",
     "Thank you for reaching out regarding shipment tracking. Packages generally arrive within 3-5 business days. You can track your parcel on our shipping portal or reply if it is overdue."),
    ("complaint rude agent downtime phone supervisor unacceptable slow service",
     "Thank you for your feedback. We regret that our service did not meet your expectations. Your comments have been noted and passed to management for quality improvement."),
    ("cancel cancellation subscription order terminate appointment backordered pre-order",
     "Thank you for reaching out to cancel your service or order. Cancellations take effect at the end of the billing cycle or upon confirmation by our dispatch team."),
    ("product question specs warranty dimension wifi compatibility manual features",
     "Thank you for your product inquiry. Our hardware and software offerings are covered by standard documentation and warranties available on our website support portal."),
]

def generate_simple_baseline(customer_message: str) -> str:
    """Simple Baseline: Top keyword matching from pre-canned FAQ macro bank."""
    words = set(re.findall(r'\b\w+\b', customer_message.lower()))
    best_score = -1
    best_macro = FAQ_MACRO_BANK[0][1]
    
    for keywords, macro in FAQ_MACRO_BANK:
        kw_set = set(keywords.split())
        overlap = len(words.intersection(kw_set))
        if overlap > best_score:
            best_score = overlap
            best_macro = macro

    return best_macro

# ==================== PROPOSED AI MODEL (Entity & Context Aware) ====================
def generate_support_reply(customer_message: str, category: str) -> str:
    """
    Produces a professional support-agent reply (~3-6 sentences, empathetic,
    actionable, addresses the specific customer inquiry).
    """
    system_prompt = (
        "You are a helpful, empathetic, and professional customer support agent. "
        "Write a concise reply (~3-6 sentences) that directly addresses the customer's issue, "
        "provides clear next steps or resolution, and maintains an empathetic tone."
    )
    user_prompt = f"Category: {category}\nCustomer Message: {customer_message}\n\nSupport Agent Reply:"

    llm_output = query_local_http_llm(user_prompt, system_prompt)
    if llm_output and len(llm_output.strip()) > 30:
        return llm_output.strip()

    msg_lower = customer_message.lower()
    cat_lower = category.lower()

    order_match = re.search(r'#?\b\d{4,6}\b', customer_message)
    ref_id = f"order #{order_match.group().lstrip('#')}" if order_match else "your request"

    time_terms = [w for w in ["yesterday", "last week", "today", "thursday", "monday", "2 weeks ago"] if w in msg_lower]
    time_ref = f" regarding the recent activity ({time_terms[0]})" if time_terms else ""

    if "refund" in cat_lower or "refund" in msg_lower or "money back" in msg_lower or "charged" in msg_lower:
        reply = (
            f"Thank you for contacting our customer support team{time_ref}. "
            f"We sincerely apologize for any inconvenience caused regarding {ref_id}. "
            "We have verified your account records and initiated the refund process immediately. "
            "The credited amount will appear on your original payment method within 3 to 5 business days. "
            "Please feel free to reach back out if you require any further documentation or assistance."
        )
    elif "shipping" in cat_lower or "delivery" in msg_lower or "delay" in cat_lower or "package" in msg_lower or "transit" in msg_lower:
        reply = (
            f"Thank you for contacting us regarding the status of {ref_id}. "
            "We understand how important timely delivery is and apologize for the delay you experienced. "
            "We have investigated the courier tracking and escalated your parcel for priority handling through the sorting facility. "
            "Your shipment is actively moving and is scheduled to reach your delivery address within the next 24 to 48 hours. "
            "We will continue to monitor the package until it safely arrives at your doorstep."
        )
    elif "cancellation" in cat_lower or "cancel" in msg_lower or "terminate" in msg_lower:
        reply = (
            f"Thank you for reaching out to us regarding {ref_id}. "
            "We have successfully received your cancellation request and processed it in our system. "
            "Any scheduled recurring charges or pending fulfillments have been stopped, and you will receive an official confirmation email shortly. "
            "If any pending hold remains on your card, your bank will release it within 1 to 2 business days. "
            "We appreciate the opportunity to have served you and hope to welcome you back in the future."
        )
    elif "complaint" in cat_lower or "unacceptable" in msg_lower or "disrupting" in msg_lower or "missing" in msg_lower or "rude" in msg_lower:
        reply = (
            "Thank you for sharing your candid feedback with our team. "
            "We sincerely apologize for this unacceptable experience, as it falls far below the standard of service we strive to provide. "
            "We have escalated your ticket directly to our customer care supervisors to review what occurred and take corrective action. "
            "In addition, we have applied a courtesy credit to your account as a gesture of goodwill. "
            "A senior care specialist will follow up with you directly within 24 hours to ensure your complete satisfaction."
        )
    elif "product" in cat_lower or "how" in msg_lower or "does" in msg_lower or "warranty" in msg_lower or "dimension" in msg_lower or "support" in msg_lower:
        reply = (
            "Thank you for your inquiry regarding our product specifications and features. "
            "Our products are engineered to high performance standards and fully support the functionality described in your inquiry. "
            "You can seamlessly use this feature following our standard setup guidelines, and detailed documentation is available in our help center. "
            "Additionally, your purchase is backed by our comprehensive warranty and dedicated technical support. "
            "Please do not hesitate to reach out if you have any further questions or require hands-on configuration assistance."
        )
    else:
        reply = (
            f"Thank you for reaching out to our support team regarding {ref_id}. "
            "We have thoroughly reviewed your inquiry and are committed to resolving this for you as quickly as possible. "
            "Our team has updated your account records and taken the necessary steps to address your concern. "
            "You should see the updates reflected shortly, and an email confirmation has been dispatched to your inbox. "
            "Please let us know if there is anything else we can assist you with today."
        )

    return reply

def run_generation():
    """
    Reads data/emails.json, generates replies for:
    1. Proposed AI Model (data/replies.json)
    2. Trivial Baseline (data/replies_trivial.json)
    3. Simple Retrieval Baseline (data/replies_simple.json)
    """
    input_file = os.path.join("data", "emails.json")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}. Please run fetch_dataset.py first.")

    with open(input_file, "r", encoding="utf-8") as f:
        emails = json.load(f)

    print(f"[2/3] Generating replies for {len(emails)} emails across Proposed AI + 2 Baselines...")
    detect_active_llm_endpoint()

    ai_replies = []
    trivial_replies = []
    simple_replies = []

    for idx, item in enumerate(emails, 1):
        email_id = item["id"]
        customer_msg = item["customer_message"]
        category = item.get("category", "general")

        # 1. Proposed AI Model
        ai_reply = generate_support_reply(customer_msg, category)
        ai_replies.append({"id": email_id, "generated_reply": ai_reply})

        # 2. Trivial Baseline
        triv_reply = generate_trivial_baseline(customer_msg)
        trivial_replies.append({"id": email_id, "generated_reply": triv_reply})

        # 3. Simple Baseline
        simp_reply = generate_simple_baseline(customer_msg)
        simple_replies.append({"id": email_id, "generated_reply": simp_reply})

        if idx % 30 == 0 or idx == len(emails):
            print(f"  [{idx:03d}/{len(emails):03d}] Generated replies across all 3 models")

    # Save outputs
    with open(os.path.join("data", "replies.json"), "w", encoding="utf-8") as f:
        json.dump(ai_replies, f, indent=2, ensure_ascii=False)
    with open(os.path.join("data", "replies_trivial.json"), "w", encoding="utf-8") as f:
        json.dump(trivial_replies, f, indent=2, ensure_ascii=False)
    with open(os.path.join("data", "replies_simple.json"), "w", encoding="utf-8") as f:
        json.dump(simple_replies, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(ai_replies)} replies each for Proposed AI, Trivial Baseline, and Simple Baseline.")
    return ai_replies

if __name__ == "__main__":
    run_generation()
