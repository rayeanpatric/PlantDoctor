import os
from groq import Groq
import gradio as gr

# Load API key from environment variable
API_KEY = "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"
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable. Set it in Hugging Face Spaces Secrets.")

client = Groq(api_key=API_KEY)

VALIDATION_PROMPT = "You are an intelligent assistant. Analyze the input question carefully. Respond with 'Yes' if the input is agriculture-related, and 'No' otherwise."
RESPONSE_PROMPT = "You are an agriculture expert. Provide a concise and accurate answer to the following agriculture-related question:"

def validate_input(input_text):
    try:
        validation_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": VALIDATION_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
            max_completion_tokens=1,
        )
        return validation_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def get_agriculture_response(input_text):
    try:
        detailed_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RESPONSE_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.5,
            max_completion_tokens=700,
        )
        return detailed_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def groq_chatbot(input_text, chat_history):
    validation_result = validate_input(input_text)
    
    if validation_result.lower() == "yes":
        response = get_agriculture_response(input_text)
    elif validation_result.lower() == "no":
        response = "❌ This is not an agriculture-related question."
    else:
        response = f"⚠️ Unexpected response: {validation_result}"

    # Append to chat history
    chat_history.append((input_text, response))
    return chat_history, ""  # Clears input field after submission

def launch_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### 🌱 Agriculture AI Assistant")
        gr.Markdown("Ask questions about plant diseases, treatments, or general agricultural topics.")

        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Ask a question about agriculture...", label="Your Question")
        clear = gr.Button("Clear Chat")

        chat_history_state = gr.State([])  # Stores chat history

        # ✅ Enter key submits the question
        msg.submit(fn=groq_chatbot, inputs=[msg, chat_history_state], outputs=[chatbot, msg])

        # ✅ Clicking "Clear Chat" resets history
        clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    demo.launch(share=True)

if __name__ == "__main__":
    launch_gradio_interface()
