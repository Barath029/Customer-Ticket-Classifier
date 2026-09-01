def generate_suggested_response(category: str, priority: str, sentiment: str, department: str, ticket_text: str = "") -> str:
    """
    Generates a tailored, empathetic, and actionable support agent response 
    based on ticket metadata predictions.
    """
    cat = str(category).lower()
    pri = str(priority).lower()
    sen = str(sentiment).lower()
    dept = str(department).lower()
    
    # 1. Opening greeting and empathy hook based on sentiment
    if sen == "negative":
        greeting = "Dear Customer, thank you for reaching out. We sincerely apologize for the frustration and inconvenience this issue has caused."
    elif sen == "positive":
        greeting = "Dear Customer, thank you for contacting us! We really appreciate your kind words and positive feedback."
    else:
        greeting = "Dear Customer, thank you for contacting support. We have received your query and are here to help."
        
    # 2. Urgency note based on priority
    if pri in ["urgent", "high"]:
        urgency_note = f" Due to the high priority of your request, this ticket has been flagged for expedited handling by our {department} team."
    else:
        urgency_note = f" Your request is currently queued for review by our {department} team."

    # 3. Action plan based on Category
    if "payment" in cat or "refund" in cat:
        action_plan = (
            "Our Billing specialists are verifying your transaction records. "
            "If a duplicate charge or payment failure occurred, any pending authorizations or eligible refunds "
            "will be processed back to your original payment method within 3 to 5 business days."
        )
    elif "account" in cat or "login" in cat:
        action_plan = (
            "Our Technical Support engineers are resetting your access parameters. "
            "Please check your inbox (and spam folder) for a secure password reset link or verification code."
        )
    elif "delivery" in cat or "order" in cat:
        action_plan = (
            "Our Logistics team is checking with our carrier partner regarding shipment status and real-time tracking updates. "
            "We will send a revised estimated delivery window shortly."
        )
    elif "product" in cat:
        action_plan = (
            "Our Product Support specialists are investigating your device/item specifications. "
            "If your item is defective, we will gladly arrange a free replacement or warranty service."
        )
    elif "technical" in cat:
        action_plan = (
            "Our Technical Support team is reviewing application error logs to isolate the root cause. "
            "Please ensure you are running the latest software build while we investigate."
        )
    elif "cancellation" in cat or "subscription" in cat:
        action_plan = (
            "Our Customer Success team is updating your subscription parameters. "
            "You will receive written confirmation once your account changes or plan cancellation are finalized."
        )
    elif "security" in cat or "fraud" in cat:
        action_plan = (
            "CRITICAL ALERT: Our Security team has initiated a protective lock and fraud investigation on your account. "
            "Please do not share your credentials with anyone while we verify account activity."
        )
    else:
        action_plan = f"Our {department} team is working on resolving your query promptly."

    # 4. Closing sign-off
    closing = "\n\nWarm regards,\nCustomer Care Operations Team"
    
    return f"{greeting}{urgency_note}\n\n{action_plan}{closing}"

if __name__ == "__main__":
    sample_response = generate_suggested_response("Payment", "High", "Negative", "Billing")
    print("Sample Suggested AI Response:\n", sample_response)
