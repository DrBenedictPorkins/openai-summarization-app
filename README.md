# openai-summarization-app

This Python-based project is designed to summarize text using OpenAI's GPT-3 model. It provides a user-friendly interface powered by Streamlit.

## Installation

Before running the application, you need to set up the Python environment and install the required dependencies. 
We recommend using [Poetry](https://python-poetry.org/) as your package manager.

1. **Clone the repository:**

   ```shell
   git clone https://github.com/yourusername/openai-summarization-app.git
   cd openai-summarization-app

2. **Install Poetry (if not already installed):**

    ```shell
    pip install poetry 
3. Set up environment variables:

    ```shell
    cp .env.example .env
    ```
    Update `.env` to include your OPENAI_API_KEY.


## Usage
```shell
   poetry run streamlit run <path to main.py>
```