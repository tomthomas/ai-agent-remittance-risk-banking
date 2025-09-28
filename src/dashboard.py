import gradio as gr
import httpx
import pandas as pd
import random
from datetime import datetime

# Simulated transaction stream (replace with Supabase query later)
def generate_transaction():
    is_high_risk = random.random() < 0.1
    return {
        "sender_country": random.choice(["UAE", "USA", "UK"]),
        "receiver_country": "Sudan" if is_high_risk else random.choice(["Nigeria", "India", "Sudan"]),
        "amount": 100000.0 if is_high_risk else random.uniform(1000, 50000),
        "currency": "USD" if is_high_risk else random.choice(["NGN", "USD", "GBP"]),
        "remittance_purpose": "Unknown" if is_high_risk else random.choice(["Family Support", "Business"]),
        "payment_method": "Cash" if is_high_risk else random.choice(["Bank Transfer", "Cash"]),
        "transaction_status": "Completed",
        "bank": random.choice(["Emirates NBD", "HSBC"]),
        "agent": "MoneyGram" if is_high_risk else random.choice(["Western Union", "MoneyGram"]),
        "transaction_type": "Online",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Initialize transaction list
def init_transactions():
    return []

# Fixed chatbot function - let Gradio handle the message format
def chat_function(message, history):
    try:
        # Convert Gradio's history format to the format your API expects
        # history is a list of [user_msg, bot_msg] pairs
        formatted_history = []
        for user_msg, bot_msg in history:
            formatted_history.append({"role": "user", "content": str(user_msg)})
            if bot_msg:  # bot_msg might be None during streaming
                formatted_history.append({"role": "assistant", "content": str(bot_msg)})
        
        # Add current message
        formatted_history.append({"role": "user", "content": str(message)})
        
        # Call your API
        with httpx.Client() as client:
            response = client.post(
                "http://127.0.0.1:8000/chat",
                json={"text": str(message), "history": formatted_history},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
        
        # Return just the response text - Gradio will handle adding it to history
        return result["response"]
    except Exception as e:
        return f"Error: {str(e)}"

# Update live transactions
def update_table(transactions):
    if not transactions:
        transactions = []
    if len(transactions) >= 10:
        transactions.pop(0)  # Keep max 10
    transaction = generate_transaction()
    try:
        with httpx.Client() as client:
            response = client.post("http://127.0.0.1:8000/predict", json=transaction, timeout=30)
            response.raise_for_status()
            result = response.json()
        prob = result["combined_probability"] * 100
        transactions.append({
            "Time": transaction["timestamp"],
            "Amount": f"${transaction['amount']:.2f}",
            "Route": f"{transaction['sender_country']} to {transaction['receiver_country']}",
            "Fraud Probability": f"{prob:.1f}%"
        })
        alert = f"Alert: High-risk transaction detected ({prob:.1f}%)!" if prob > 70 else ""
        df = pd.DataFrame(transactions)
        print(f"Table updated: {len(transactions)} rows, Alert: {alert}, Data: {df.to_dict()}")  # Debug log
        return df, alert, transactions
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Table update error: {error_msg}")  # Debug log
        return pd.DataFrame(transactions), error_msg, transactions

# Gradio interface
with gr.Blocks(css=".red {color: red; font-weight: bold;}") as demo:
    gr.Markdown("# Remittance Risk Agent Dashboard")
    transactions = gr.State(init_transactions)
    
    with gr.Tab("Chatbot"):
        gr.Markdown("### Chat with Risk Agent")
        gr.Markdown("Ask about a transaction (e.g., 'Send $25000 from UAE to Nigeria for family support').")
        chatbot = gr.ChatInterface(
            fn=chat_function,
            examples=["Send $25000 from UAE to Nigeria for family support", "UAE to India", "hello"],
            title="Risk Assessment Chatbot"
            # Removed type="messages" - let Gradio use default format
        )
    
    with gr.Tab("Live Transactions"):
        gr.Markdown("### Real-Time Transaction Monitor")
        table_output = gr.DataFrame(headers=["Time", "Amount", "Route", "Fraud Probability"])
        alert_output = gr.Textbox(label="Alerts")
        gr.Timer(value=2.0).tick(update_table, inputs=transactions, outputs=[table_output, alert_output, transactions])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)