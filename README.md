# Web Content Q&A Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://website-interaction.streamlit.app/)

**[Deployed Application Link: https://website-interaction.streamlit.app/](https://website-interaction.streamlit.app/)**

## Overview

This is a simple web-based tool built with Streamlit, Langchain, and Google Gemini API that allows you to ask questions about the content of websites you provide.  The tool is designed to answer questions *solely* based on the information scraped from the URLs you input, without relying on general world knowledge.

**Key Features:**

*   **URL Input:**  Users can enter one or more website URLs in a text area.
*   **Content Ingestion:**  The tool scrapes the text content from the provided URLs. It also supports ingesting content from `sitemap.xml` files for broader website coverage.
*   **Question Answering:** Users can ask questions related to the ingested website content.
*   **Accurate Answers:** Answers are generated using Google's Gemini Pro model and are grounded strictly in the scraped website content.
*   **Simple UI:**  A user-friendly Streamlit interface with clear input fields and buttons.
*   **Two Ingestion Modes:**
    *   **Ingest URLs:** Processes content from the URLs directly entered by the user.
    *   **Ingest all subdomains:**  Attempts to find and process content from the `sitemap.xml` of each provided URL, potentially covering more pages of the website.
*   **Persistent Vector Store:** The ingested website content is vectorized and stored, allowing you to ask multiple questions without re-ingesting the URLs each time.

## Evaluation Criteria

This project was built with the following evaluation criteria in mind:

*   **Relevance & Accuracy of answers:**  Answers should be directly relevant to the ingested website content and factually accurate based on that content alone.
*   **UI/UX:** The user interface should be straightforward, intuitive, and easy to use for anyone.
*   **Implementation Clarity:** The codebase should be well-organized, commented, and maintainable for future modifications or understanding.

## How to Run Locally

Follow these steps to run the Web Content Q&A Tool on your local machine:

1.  **Prerequisites:**
    *   **Python 3.8 or higher** must be installed on your system.
    *   **Pip** (Python package installer) should be installed.

2.  **Install Python Libraries:**
    Open your terminal or command prompt and run the following command to install the necessary Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Get a Google AI Studio API Key:**
    *   Go to [Google AI Studio](https://makersuite.google.com/app/apikey) and create a project.
    *   Generate an API key for the Gemini API within your project.
    *   **Important Security Note:**  For local testing, you will enter this API key directly into the application's text box. **This is NOT a secure method for production deployments.**

4.  **Run the Streamlit App:**
    Navigate to the directory where you saved the `app.py` file in your terminal. Run the Streamlit application using the command:

    ```bash
    streamlit run app.py
    ```

5.  **Access the App in Your Browser:**
    Streamlit will provide a local URL in your terminal (usually `http://localhost:8501`). Open this URL in your web browser to access the Web Content Q&A Tool.

    *   **Enter your Google AI Studio API key** into the provided text box.
    *   **Enter the website URLs** you want to query (one URL per line).
    *   Click either **"Ingest URLs"** or **"Ingest all subdomains"** to process the website content.
    *   **Ask your question** in the question input box.
    *   Click **"Ask Question"** to get your answer.

## Deployment (and Security Considerations)

**Warning: Entering your API key directly into the code is highly insecure and is only recommended for local testing.**  **Do not use this method for production or publicly accessible deployments.**

For secure deployment, especially if you are using Streamlit Cloud or other platforms, you should use secure methods to manage your API keys, such as:

*   **Streamlit Secrets (Recommended for Streamlit Cloud):**
    1.  In your Streamlit Cloud app settings, define a secret named `GOOGLE_API_KEY` and paste your actual API key as the value.
    2.  In your `app.py` code, uncomment the API key input text box or replace it with the following line to load the API key from secrets:

        ```python
        api_key = st.secrets["GOOGLE_API_KEY"]
        ```
        And revert the `initialize_llm`, `initialize_embeddings`, and `ingest_urls` functions back to using `GOOGLE_API_KEY` directly instead of passing it as an argument.
    3.  Deploy your application to Streamlit Cloud.

*   **Environment Variables (For other hosting platforms):** Configure your hosting environment to set an environment variable named `GOOGLE_API_KEY` with your API key value. Access it in your Python code using `os.environ.get("GOOGLE_API_KEY")`.

**Steps for Streamlit Cloud Deployment (using Secrets):**

1.  **Push your code to a GitHub repository.**
2.  **Sign up for Streamlit Cloud** at [streamlit.io/cloud](https://streamlit.io/cloud).
3.  **Connect your GitHub repository to Streamlit Cloud.**
4.  **Set up your API Key as a Secret in Streamlit Cloud:** In your Streamlit Cloud app's settings, add a secret named `GOOGLE_API_KEY` and paste your Gemini API key as the value.
5.  **Revert your code/uncomment the st.secrets part to use `st.secrets["GOOGLE_API_KEY"]`** for secure API key loading (as mentioned above).
6.  **Deploy your app from your GitHub repository in Streamlit Cloud.**

## Source Code

[Link to your GitHub Repository will be here]

## Note

This is a basic implementation of a Web Content Q&A Tool and can be further enhanced. Potential future improvements could include:

*   More robust error handling and user feedback.
*   Improved UI/UX design.
*   More advanced text processing and chunking strategies for better content ingestion.
*   Exploration of different Langchain chain types and retrieval methods for optimized question answering.
*   Support for different document loaders and file types.
*   Addition of OpenAI api as alternate to google gemini api

---

Feel free to contribute to this project or use it as a starting point for your own web content analysis tools!