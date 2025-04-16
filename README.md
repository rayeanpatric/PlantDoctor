---
license: mit
sdk: gradio
colorFrom: purple
colorTo: pink
title: PlantDoctor
emoji: 🌍
sdk_version: 5.25.2
app_file: app.py
---
# Plant Disease Diagnosis and Agricultural Chatbot

This application, hosted on Hugging Face Spaces, allows users to diagnose plant diseases by uploading images of plant leaves and receive treatment recommendations. It also features an AI-powered chatbot to answer agricultural queries, leveraging a pre-trained machine learning model and the Groq API.

## Features

- **Disease Diagnosis:** Upload a leaf image to identify common plant diseases using a pre-trained MobileNetV2 model.
- **Treatment Recommendations:** Get detailed treatment suggestions based on the diagnosed disease.
- **Agricultural Chatbot:** Ask questions about plant diseases, treatments, or general agricultural topics and receive AI-generated responses via the Grok API.

## Access the Application

The application is hosted on Hugging Face Spaces and can be accessed directly through this link:

[Plant Disease Diagnosis and Agricultural Chatbot on Hugging Face Spaces](https://huggingface.co/spaces/rayeanpatric/PlantDoctor)

## Usage

1. **Access the Interface:**
   - Visit the Hugging Face Space using the link above.
   - The application will load in your web browser.

2. **Diagnose Plant Disease:**
   - Go to the "Diagnose Plant Disease" tab.
   - Upload a clear, well-lit image of a plant leaf (supported formats: JPEG, PNG, etc.).
   - Click the "Diagnose Disease" button.
   - Check the diagnosis, confidence level, and treatment recommendation in the output section.

3. **Agricultural Chatbot:**
   - Switch to the "Agricultural Chatbot" tab.
   - Enter your question in the text box (e.g., "How do I treat tomato blight?").
   - Press Enter to submit.
   - Read the chatbot’s response in the chat history.
   - Use the "Clear Chat" button to start a new conversation.

**Notes:**
- **Diagnosis Limitations:** The disease diagnosis relies on the PlantVillage dataset and supports species like Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. It may not detect diseases outside this dataset.
- **Chatbot Accuracy:** Responses are AI-generated via the Grok API and may not always be accurate. Verify critical information with an agricultural expert.

## About this Application

- **Disease Diagnosis:** Powered by a MobileNetV2 model fine-tuned on the PlantVillage dataset to identify plant diseases from leaf images.
- **Chatbot:** Utilizes the Grok API with the `llama-3.3-70b-versatile` model for dynamic question-answering.
- **Supported Plants:** Includes diagnosis for Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

## Troubleshooting

- **Image Upload Issues:** Ensure your leaf images are clear and in supported formats (JPEG, PNG).
- **Chatbot Not Responding:** Check your internet connection or refresh the page.
- **Application Not Loading:** Use a compatible browser (e.g., Chrome, Firefox) and ensure JavaScript is enabled.

## Contributing

Since the application is hosted on Hugging Face Spaces and cannot be cloned locally, contributions can be made by:
1. Forking the space on Hugging Face.
2. Implementing your changes.
3. Submitting a pull request or collaborating via Hugging Face’s platform.

For more information, check Hugging Face’s documentation on collaborating with spaces.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.