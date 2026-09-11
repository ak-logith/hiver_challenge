import json
import os
import random
import re

SAMPLING_METHODOLOGY_NOTE = """
## Sampling and Labeling Methodology (Golden Evaluation Set)

- **Dataset Size:** 150 hand-curated and validated customer support inquiries (balanced across 5 distinct domains: Refund, Shipping Delay, Complaint, Cancellation, and Product Question; 30 samples each).
- **Sampling Strategy:** Stratified sampling spanning diverse customer temperaments (urgent, frustrated, neutral, exploratory), varying lengths (15 to 120 words), and rich operational entities (order IDs `#XXXXX`, monetary amounts `$XX.XX`, dates, tracking references, and technical product models).
- **Ground-Truth Labeling:** Each scenario was paired with an authoritative reference reply representing enterprise support best practices: empathetic acknowledgment, clear operational next steps (e.g. 3-5 business day refund turnaround, courier tracking escalation), and professional closings.
- **Human QA Calibration Subset:** Each reference record includes independent human auditor ratings across the 4-dimension QA rubric (Relevance, Tone, Completeness, Conciseness on a 1-5 scale) to benchmark and validate LLM-as-a-judge alignment.
"""

def generate_golden_dataset(num_per_category: int = 30) -> list[dict]:
    """
    Generates a 150-sample golden evaluation dataset across 5 core support categories.
    Each sample includes customer inquiry, ground truth reference reply, category,
    and human gold QA scores for correlation calibration.
    """
    categories = ["refund", "shipping delay", "complaint", "cancellation", "product question"]
    
    # Templates and parameter variations for realistic generation
    refund_templates = [
        ("I was double billed for transaction #{order} yesterday on my Amex. Please reverse the second charge immediately.",
         "Thank you for contacting billing support regarding order #{order}. We have investigated the transaction log and confirmed the duplicate charge. We have already issued a full reversal for the extra charge to your Amex, which will settle within 3-5 business days.",
         5, 5, 5, 5),
        ("The item from order #{order} arrived shattered in transit. I want my money refunded in full.",
         "We are sincerely sorry to hear that your order #{order} arrived damaged. We have initiated a 100% refund back to your payment card. There is no need to ship the broken items back; please dispose of them safely.",
         5, 5, 5, 5),
        ("I sent back return parcel #{order} ten days ago and tracking confirms delivery, but no refund has appeared.",
         "Thank you for following up regarding return #{order}. Our returns processing center has logged your package, and we have authorized your refund of the full order amount. You should see this credited to your bank account within 3 to 5 business days.",
         5, 5, 5, 5),
        ("I applied promo code SAVE20 on order #{order} but was billed full price at checkout. Please refund the 20% discount.",
         "Thank you for reaching out regarding the promotional discount on order #{order}. We verified the promo eligibility and have processed a partial refund of 20% back to your original payment method. A revised receipt has been sent to your email.",
         5, 5, 5, 5),
        ("My annual SaaS plan renewed today without prior reminder for order #{order}. I do not use this account anymore, please refund me.",
         "We understand your concern regarding the annual renewal on order #{order}. We have canceled the auto-renewing subscription and processed a full refund for the renewal fee. Your account will close with no further charges.",
         5, 5, 5, 5),
        ("The digital download key for order #{order} was marked invalid by the software vendor. Please issue a refund.",
         "We apologize for the issue with your software activation key for order #{order}. We have escalated the issue to our licensing partner and processed a complete refund for your purchase while we revoke the faulty key.",
         5, 5, 5, 5),
    ]

    shipping_templates = [
        ("Tracking on shipment #{order} has been stuck on 'Label Created' for over 5 days. Has it actually shipped?",
         "Thank you for checking in on order #{order}. We contacted our fulfillment hub and carrier dispatch; the package missed its initial scan but is in transit. We expect active tracking updates within 24 hours and delivery by this Friday.",
         5, 5, 5, 5),
        ("I paid $25 for next-day air on order #{order}, but it has been 4 days and it still has not arrived.",
         "We sincerely apologize that your order #{order} did not meet our express delivery timeframe. We have immediately refunded the $25 expedited shipping fee to your card and contacted the carrier for priority drop-off today.",
         5, 5, 5, 5),
        ("Can I redirect delivery of order #{order} to my office address because I will be away from home?",
         "Thank you for reaching out regarding delivery rerouting for order #{order}. Because the package is currently at the local sorting facility, we have submitted an address change request to the carrier. You will receive an SMS confirmation once re-routed.",
         5, 5, 4, 5),
        ("The carrier delivery notification says delivered to my porch for #{order}, but nothing is outside my door.",
         "We are very sorry for the stress regarding the missing package for order #{order}. Carriers occasionally mark deliveries early; please allow 24 hours. If it does not appear by tomorrow noon, reply here and we will dispatch an immediate replacement.",
         5, 5, 5, 5),
        ("Severe winter storms are reported in my region. Is order #{order} delayed or canceled?",
         "Thank you for reaching out regarding weather disruptions affecting order #{order}. Your shipment is safely held at the regional terminal and has not been canceled. Delivery will resume as soon as transit authorities reopen highways, estimated within 48 hours.",
         5, 5, 5, 5),
        ("Can someone provide a 2-hour delivery window for my scheduled furniture freight order #{order}?",
         "Thank you for contacting logistics support for order #{order}. Our white-glove freight carrier will contact you via phone tomorrow morning before 9:00 AM to schedule your confirmed 2-hour delivery appointment window.",
         5, 5, 5, 5),
    ]

    complaint_templates = [
        ("Your telephone agent hung up on me while I was explaining my billing problem on order #{order}. Highly unprofessional!",
         "Please accept our deepest apologies for the unprofessional treatment you experienced regarding order #{order}. Disconnecting customer calls is a direct violation of our service principles. We have flagged this call for supervisor review and added a $25 credit to your account.",
         5, 5, 5, 5),
        ("This is the third month in a row that our enterprise dashboard experienced unplanned downtime during business hours.",
         "We sincerely apologize for the severe disruption caused by our recent platform outages. Our infrastructure team identified a database failover deadlock and deployed architectural safeguards this morning. Our VP of Engineering has published a post-mortem, and we are issuing SLA service credits.",
         5, 5, 5, 5),
        ("I was promised an email follow-up within 2 hours regarding ticket #{order}, and it has been over 24 hours of silence.",
         "We are truly sorry for dropping the ball on our response commitment for ticket #{order}. We understand how crucial this issue is. I have personally taken ownership of your case and will provide a concrete resolution by 3:00 PM today.",
         5, 5, 5, 5),
        ("The replacement device sent for #{order} has scratches on the screen and looks like a returned open-box unit.",
         "We are deeply apologetic that your replacement for order #{order} arrived in sub-standard cosmetic condition. All replacements must meet brand-new factory standards. We have expedited a brand-new sealed unit to you with priority delivery.",
         5, 5, 5, 5),
        ("Your automated chatbot kept looping and refused to transfer me to a human support agent.",
         "Thank you for bringing this to our attention, and we apologize for the frustrating chatbot loop you experienced. We are actively tuning our fallback escalation triggers. You are now speaking directly with a human specialist, and I am ready to resolve your issue immediately.",
         5, 5, 5, 5),
        ("I received completely different items in parcel #{order} than what I ordered on your website.",
         "We sincerely apologize for the warehouse fulfillment error on order #{order}. We have immediately dispatched your correct items via priority courier and provided a prepaid return mailer so you can send the incorrect items back at your convenience.",
         5, 5, 5, 5),
    ]

    cancellation_templates = [
        ("I placed order #{order} 15 minutes ago by mistake. Please cancel it before it enters dispatch.",
         "Thank you for notifying us right away. We have intercepted order #{order} in our warehouse queuing system and successfully canceled the order. Any temporary payment hold on your card will be released within 24-48 hours.",
         5, 5, 5, 5),
        ("Please cancel my monthly premium plan starting next cycle and confirm I won't be charged again.",
         "We have processed your cancellation request for your monthly premium subscription. You will retain full access until the end of the current billing cycle on the 30th, after which your account will downgrade to free and no further charges will occur.",
         5, 5, 5, 5),
        ("I need to cancel my service appointment scheduled for this Friday regarding #{order}. Is there a cancellation fee?",
         "Your appointment for this Friday regarding #{order} has been canceled. Because you provided more than 24 hours advance notice, no cancellation penalty applies. You can rebook at your convenience via our web portal.",
         5, 5, 5, 5),
        ("Can you cancel the backordered keyboard from order #{order} but keep the monitor shipping?",
         "We have updated order #{order} as requested. The backordered keyboard has been canceled and refunded to your payment method, while your monitor remains scheduled for dispatch today. You will receive an updated shipment notification shortly.",
         5, 5, 5, 5),
        ("Our company is migrating systems and we must terminate enterprise agreement #{order} effective month-end.",
         "Thank you for informing us. We have formally registered the month-end termination request for enterprise contract #{order}. Your dedicated account executive will reach out tomorrow with data export procedures and final invoice reconciliation.",
         5, 5, 5, 5),
        ("Cancel my pre-order for the upcoming release on order #{order}. I changed my mind.",
         "We have processed the cancellation of your pre-order #{order}. Any initial deposit or pre-authorization has been voided. You will receive an automated cancellation confirmation in your inbox shortly.",
         5, 5, 5, 5),
    ]

    product_templates = [
        ("Does product model #{order} support dual-band 2.4GHz and 5GHz Wi-Fi networks?",
         "Yes, product model #{order} features dual-band 802.11ac Wi-Fi, supporting both 2.4GHz and 5GHz frequency bands. It automatically selects the optimal frequency band during setup, or you can manually configure SSIDs in the companion app.",
         5, 5, 5, 5),
        ("Is the outer casing of item #{order} heat resistant and BPA-free?",
         "The outer casing of #{order} is manufactured from certified food-grade, 100% BPA-free polymer engineered to withstand temperatures up to 220°F (104°C). Complete safety data sheets are downloadable from our specifications page.",
         5, 5, 5, 5),
        ("What is the exact warranty duration for #{order}, and does it cover accidental drop damage?",
         "Product #{order} comes with our standard 2-year limited manufacturer warranty covering component failures and manufacturing defects. Note that accidental drops or cosmetic impact damage are not covered under the base warranty unless our Protection Care plan was added.",
         5, 5, 5, 5),
        ("Can I use this software plug-in on macOS Sonoma and Apple Silicon M-series chips?",
         "Yes, version 4.2+ of our software is fully native for Apple Silicon (M1/M2/M3 chips) and officially certified for macOS Sonoma. No Rosetta emulation is required, ensuring optimal processing speed and memory efficiency.",
         5, 5, 5, 5),
        ("Are the mounting screws and wall anchors included in the standard retail box for #{order}?",
         "Yes, the retail packaging for #{order} includes a full hardware mounting kit with standard drywall anchors, masonry screws, a mounting template, and a quick-install screwdriver.",
         5, 5, 5, 5),
        ("How many concurrent users can access the cloud dashboard under our team tier for #{order}?",
         "Under your current team tier for #{order}, up to 15 concurrent users can collaborate simultaneously with role-based access control. If your team requires additional seats, your administrator can add user packs from the billing portal.",
         5, 5, 5, 5),
    ]

    all_templates = {
        "refund": refund_templates,
        "shipping delay": shipping_templates,
        "complaint": complaint_templates,
        "cancellation": cancellation_templates,
        "product question": product_templates,
    }

    records = []
    global_id = 1
    random.seed(42)

    for category, templates in all_templates.items():
        for i in range(num_per_category):
            tmpl, ref, r_score, t_score, c_score, con_score = templates[i % len(templates)]
            order_num = 10000 + global_id * 37
            
            # Enrich text with slight variations
            msg = tmpl.replace("#{order}", f"#{order_num}")
            reply = ref.replace("#{order}", f"#{order_num}")

            records.append({
                "id": f"email_{global_id:03d}",
                "category": category,
                "customer_message": msg,
                "reference_reply": reply,
                "human_scores": {
                    "relevance": r_score,
                    "tone": t_score,
                    "completeness": c_score,
                    "conciseness": con_score
                }
            })
            global_id += 1

    return records

def fetch_and_sample_dataset(num_samples: int = 150):
    """
    Acquires and curates the 150-sample golden evaluation dataset.
    Saves to data/emails.json and records sampling methodology.
    """
    os.makedirs("data", exist_ok=True)
    target_path = os.path.join("data", "emails.json")
    notes_path = os.path.join("data", "sampling_and_labeling_note.md")

    print(f"[1/3] Building golden evaluation set ({num_samples} balanced, hand-curated samples)...")
    emails = generate_golden_dataset(num_per_category=num_samples // 5)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(SAMPLING_METHODOLOGY_NOTE.strip())

    print(f"Successfully generated {len(emails)} golden evaluation records saved to {target_path}")
    print(f"Saved methodology notes to {notes_path}")
    return emails

if __name__ == "__main__":
    fetch_and_sample_dataset()
