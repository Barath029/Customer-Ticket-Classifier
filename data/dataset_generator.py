import os
import random
import pandas as pd
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Categories (10)
CATEGORIES = [
    "Account/Login",
    "Payment",
    "Refund",
    "Order",
    "Delivery",
    "Product",
    "Technical Support",
    "Subscription",
    "Cancellation",
    "Security/Fraud"
]

# Priorities (4)
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

# Sentiments (3)
SENTIMENTS = ["Positive", "Neutral", "Negative"]

# Departments (6)
DEPARTMENTS = [
    "Technical Support",
    "Billing",
    "Logistics",
    "Product Support",
    "Customer Success",
    "Security"
]

# Department mapping strategy
CATEGORY_DEPARTMENT_MAP = {
    "Account/Login": "Technical Support",
    "Payment": "Billing",
    "Refund": "Billing",
    "Order": "Logistics",
    "Delivery": "Logistics",
    "Product": "Product Support",
    "Technical Support": "Technical Support",
    "Subscription": "Customer Success",
    "Cancellation": "Customer Success",
    "Security/Fraud": "Security"
}

# Template pools for realistic ticket generation
TEMPLATES = {
    "Account/Login": [
        ("I forgot my password and cannot access my account despite multiple reset attempts.", "Medium", "Negative"),
        ("Locked out of my profile after updating two-factor authentication.", "High", "Negative"),
        ("How can I change my registered email address on the account settings page?", "Low", "Neutral"),
        ("Unable to log in using Google OAuth. It throws a 500 server error.", "High", "Negative"),
        ("Thank you for helping me recover my login credentials yesterday!", "Low", "Positive"),
        ("My account was disabled suddenly without any notification.", "Urgent", "Negative"),
        ("Can I add multiple user profiles under a single master account?", "Low", "Neutral"),
        ("Keep getting invalid credentials error even though my password is correct.", "Medium", "Negative"),
        ("Reset password link sent to my email is expired or broken.", "Medium", "Negative"),
        ("Smooth login experience, just wanted to check if biometric login is coming soon.", "Low", "Positive"),
        ("Session keeps timing out every 2 minutes while working on dashboard.", "High", "Negative"),
        ("Cannot verify my phone number during account signup.", "Medium", "Negative"),
    ],
    "Payment": [
        ("I was charged twice for order #{order_id} on my credit card statement.", "High", "Negative"),
        ("Payment failed during checkout but the money was deducted from my bank account.", "Urgent", "Negative"),
        ("What payment methods do you accept for international orders?", "Low", "Neutral"),
        ("My credit card transaction was declined repeatedly without clear reason.", "High", "Negative"),
        ("Invoice #{order_id} shows incorrect tax billing details.", "Medium", "Negative"),
        ("Autopay processed successfully and invoice was clear, thank you.", "Low", "Positive"),
        ("Received a billing error notification during checkout completion.", "High", "Negative"),
        ("Is it possible to split payment between two different credit cards?", "Low", "Neutral"),
        ("Card authorized but payment status still shows pending after 24 hours.", "High", "Negative"),
        ("Overcharged on shipping fee during invoice calculation.", "Medium", "Negative"),
        ("Your payment gateway is extremely smooth and fast, great job!", "Low", "Positive"),
        ("Unrecognized charge of $79.99 appeared on my bank statement today.", "Urgent", "Negative"),
    ],
    "Refund": [
        ("I returned item #{order_id} two weeks ago and still haven't received my refund.", "High", "Negative"),
        ("Requesting a full refund for damaged product delivered yesterday.", "High", "Negative"),
        ("How long does it take for a refund to reflect in my bank account?", "Low", "Neutral"),
        ("Refund status says processed but bank confirms no credit received yet.", "High", "Negative"),
        ("Extremely disappointed with delay in processing my partial refund request.", "High", "Negative"),
        ("Received my store credit refund promptly, appreciate the swift assistance!", "Low", "Positive"),
        ("Need refund for canceled item order #{order_id}.", "Medium", "Neutral"),
        ("Returned defective gadget, please initiate immediate refund to original payment source.", "High", "Negative"),
        ("Why was my refund request rejected when product was returned unopened?", "Urgent", "Negative"),
        ("Thank you for issuing the full refund without any hassle.", "Low", "Positive"),
        ("Still waiting on refund reference number for tracking purpose.", "Medium", "Neutral"),
        ("Wrong item delivered and refund process is taking way too long.", "High", "Negative"),
    ],
    "Order": [
        ("Where can I track my order #{order_id}? Tracking number isn't updating.", "Medium", "Neutral"),
        ("I want to modify the shipping address for order #{order_id}.", "High", "Neutral"),
        ("Order status has been stuck on 'Processing' for over 5 days.", "High", "Negative"),
        ("Can I add one more item to my active order before it ships?", "Low", "Neutral"),
        ("My order was canceled automatically without any explanation.", "Urgent", "Negative"),
        ("Order arrived earlier than expected in excellent packaging, thank you!", "Low", "Positive"),
        ("Received confirmation email but order doesn't show up in my purchase history.", "Medium", "Negative"),
        ("Need an official tax invoice generated for order #{order_id}.", "Low", "Neutral"),
        ("Wrong color ordered by mistake, how can I modify item specifications?", "Medium", "Neutral"),
        ("Order summary shows incorrect quantities for item list.", "Medium", "Negative"),
        ("Order placement was effortless and checkout process was intuitive.", "Low", "Positive"),
        ("Items missing from my delivered package for order #{order_id}.", "Urgent", "Negative"),
    ],
    "Delivery": [
        ("Package #{order_id} shows delivered but I have not received it.", "Urgent", "Negative"),
        ("Courier driver left package in heavy rain and box is completely soaked.", "High", "Negative"),
        ("When is the expected delivery date for shipment tracking #{order_id}?", "Low", "Neutral"),
        ("Delivery driver missed scheduled slot and marked recipient unavailable.", "Medium", "Negative"),
        ("Package arrived damaged with broken outer box and missing contents.", "Urgent", "Negative"),
        ("Delivery executive was polite and delivered right on time!", "Low", "Positive"),
        ("Address updated on portal but courier tried delivering to old address.", "High", "Negative"),
        ("Express shipping paid but package took 7 days to arrive.", "High", "Negative"),
        ("Can I schedule weekend delivery for my upcoming package?", "Low", "Neutral"),
        ("Shipment delayed due to transit exception without revised ETA.", "Medium", "Negative"),
        ("Outstanding delivery service, package arrived in pristine condition!", "Low", "Positive"),
        ("Wrong shipping tracking link provided in notification email.", "Low", "Neutral"),
    ],
    "Product": [
        ("The product stopped working after 2 days of minimal usage.", "High", "Negative"),
        ("Product specifications on website do not match the physical item received.", "High", "Negative"),
        ("Are there replacement parts available for model #{order_id}?", "Low", "Neutral"),
        ("Item build quality is cheap and plastic handle broke immediately.", "High", "Negative"),
        ("How do I activate the 1-year manufacturer warranty for this item?", "Low", "Neutral"),
        ("The product performance exceeded my expectations! Super satisfied.", "Low", "Positive"),
        ("Received product with missing user manual and power adapter cable.", "Medium", "Negative"),
        ("Device overheating excessively while operating on standard settings.", "Urgent", "Negative"),
        ("Is this product compatible with MacOS Monterey?", "Low", "Neutral"),
        ("Buttons on the controller are non-responsive straight out of the box.", "High", "Negative"),
        ("Fantastic build quality and sleek design, highly recommend this product!", "Low", "Positive"),
        ("Screen has multiple dead pixels upon initial unboxing.", "High", "Negative"),
    ],
    "Technical Support": [
        ("Mobile application crashes every time I tap on the settings menu.", "High", "Negative"),
        ("API connection returning 504 Gateway Timeout on bulk endpoint.", "Urgent", "Negative"),
        ("How do I integrate your webhook with our internal CRM workflow?", "Low", "Neutral"),
        ("Desktop app fails to sync offline data when reconnected to internet.", "High", "Negative"),
        ("Getting database connection error during report export.", "Urgent", "Negative"),
        ("Tech support team resolved my software issue in 10 minutes flat, amazing service!", "Low", "Positive"),
        ("Feature request: support dark mode toggle on web portal.", "Low", "Neutral"),
        ("SSL certificate error throwing warning on browser load.", "High", "Negative"),
        ("Memory leak causing browser tab to freeze after 30 minutes of use.", "High", "Negative"),
        ("Video playback player stutters and buffers continuously on HD quality.", "Medium", "Negative"),
        ("Super clear documentation made technical integration seamless!", "Low", "Positive"),
        ("Unable to install latest software update patch on Windows 11.", "Medium", "Negative"),
    ],
    "Subscription": [
        ("My monthly subscription renewed automatically but I didn't receive account credits.", "High", "Negative"),
        ("How do I upgrade from Basic to Enterprise tier subscription?", "Low", "Neutral"),
        ("Want to pause my recurring subscription for 2 months while traveling.", "Low", "Neutral"),
        ("Charged annual fee instead of monthly plan selected during checkout.", "High", "Negative"),
        ("Promo discount code did not apply to subscription renewal.", "Medium", "Negative"),
        ("Subscription upgrade was instant and tier features are worth every penny!", "Low", "Positive"),
        ("What features are included in the Pro tier plan?", "Low", "Neutral"),
        ("Subscription status changed to inactive despite successful payment receipt.", "High", "Negative"),
        ("Requesting billing cycle shift from 1st of month to 15th.", "Low", "Neutral"),
        ("Accidentally subscribed twice under two different email addresses.", "Medium", "Negative"),
        ("Loved the free trial experience, excited to be on paid subscription!", "Low", "Positive"),
        ("Price hike announced without 30 days prior notice to subscribers.", "Medium", "Negative"),
    ],
    "Cancellation": [
        ("I want to cancel my active subscription immediately and avoid future charges.", "High", "Neutral"),
        ("Cancellation button on dashboard is hidden or broken.", "High", "Negative"),
        ("Was charged renewal fee after submitting cancellation request 3 days ago.", "Urgent", "Negative"),
        ("What is your cancellation policy for annual membership contracts?", "Low", "Neutral"),
        ("Please confirm cancellation of order #{order_id} before dispatch.", "High", "Neutral"),
        ("Cancellation was processed swiftly without pushy retention calls, thank you.", "Low", "Positive"),
        ("Can I cancel my order within 1 hour of placing it online?", "Low", "Neutral"),
        ("Tried canceling via chat but bot kept looping without completing request.", "Medium", "Negative"),
        ("Need written confirmation of account closure and data removal.", "Medium", "Neutral"),
        ("Canceled service due to moving abroad, smooth procedure.", "Low", "Neutral"),
        ("Extremely difficult cancellation workflow designed to trap customers.", "High", "Negative"),
        ("Confirming that my subscription will auto-terminate at end of current cycle.", "Low", "Neutral"),
    ],
    "Security/Fraud": [
        ("Suspicious unauthorized transaction of $450 detected on my account.", "Urgent", "Negative"),
        ("Received a phishing email claiming to be from your security department.", "High", "Negative"),
        ("Someone accessed my account from an unknown IP address in another country.", "Urgent", "Negative"),
        ("My password was leaked in a public data breach, need immediate account lock.", "Urgent", "Negative"),
        ("App asking for unnecessary permissions on Android system.", "Medium", "Negative"),
        ("Security response team acted instantly to secure my compromised account, heroic work!", "Low", "Positive"),
        ("How do I enable hardware key two-factor security authentication?", "Low", "Neutral"),
        ("Unrecognized API key created under my developer profile credentials.", "Urgent", "Negative"),
        ("Fake store website impersonating your brand to scam customers.", "High", "Negative"),
        ("Credit card credentials compromised right after purchase.", "Urgent", "Negative"),
        ("Appreciate your robust encryption and proactive fraud alerts!", "Low", "Positive"),
        ("Notification received for password change that I did not initiate.", "Urgent", "Negative"),
    ]
}

# Modifiers & Fillers for Synthetic Variance
PREFIXES = [
    "", "Hello support team, ", "Urgent: ", "Hi there, ", "Dear Support, ", 
    "Please help: ", "Attn Customer Care, ", "FYI: ", "Query: ", "Issue report: "
]

SUFFIXES = [
    "", " Please resolve this as soon as possible.", " Looking forward to your prompt response.",
    " Thanks for your help.", " This is very frustrating.", " Appreciate your assistance.",
    " Please advise on next steps.", " Kindly check and revert back.", " System ref ID #9921."
]

TYPOS = [
    ("password", "pasword"), ("account", "acount"), ("payment", "paymnt"),
    ("received", "recieved"), ("refund", "refnd"), ("service", "servce"),
    ("immediately", "immediatly"), ("charged", "chagred"), ("delivery", "delivry")
]

def add_noise(text, noise_prob=0.15):
    """Add subtle realistic variations and occasional typos."""
    if random.random() < noise_prob:
        target_word, typo_word = random.choice(TYPOS)
        text = text.replace(target_word, typo_word, 1)
    return text

def generate_dataset(num_samples=4200):
    records = []
    
    samples_per_category = num_samples // len(CATEGORIES)
    ticket_count = 1
    
    for category in CATEGORIES:
        templates = TEMPLATES[category]
        dept = CATEGORY_DEPARTMENT_MAP[category]
        
        for _ in range(samples_per_category):
            template_text, base_priority, base_sentiment = random.choice(templates)
            
            # Interpolate order_id if present
            order_id = f"ORD-{random.randint(10000, 99999)}"
            text = template_text.format(order_id=order_id)
            
            # Add prefix/suffix variance
            prefix = random.choice(PREFIXES) if random.random() > 0.4 else ""
            suffix = random.choice(SUFFIXES) if random.random() > 0.4 else ""
            
            full_text = add_noise(f"{prefix}{text}{suffix}")
            
            # Minor random variation in priority/sentiment for realistic overlap
            priority = base_priority
            if random.random() < 0.08:
                priority = random.choice(PRIORITIES)
                
            sentiment = base_sentiment
            if random.random() < 0.08:
                sentiment = random.choice(SENTIMENTS)

            records.append({
                "ticket_id": f"TCK-{ticket_count:05d}",
                "ticket_text": full_text,
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "department": dept
            })
            ticket_count += 1
            
    df = pd.DataFrame(records)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def save_dataset_splits(df, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, "customer_tickets.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved full dataset ({len(df)} rows) to {full_path}")
    
    # Split: 70% Train, 15% Val, 15% Test
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    df = generate_dataset(num_samples=4200)
    save_dataset_splits(df)
