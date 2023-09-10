import dotenv
import streamlit as st
import tiktoken
import openai
import collections
import re
import os
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from openai.error import OpenAIError
# from langchain.chat_models import ChatOpenAI
# from langchain.docstore.document import Document
# from langchain.chains.summarize import load_summarize_chain

dotenv.load_dotenv()
openai.api_key =os.getenv('OPENAI_API_KEY')

# MODELS = {
#     "gpt-4": 8192,
#     "gpt-4-0613": 8192,
#     "gpt-4-32k": 32768,
#     "gpt-4-32k-0613": 32768,
#     "gpt-3.5-turbo": 4096,
#     "gpt-3.5-turbo-16k": 16384,
#     "gpt-3.5-turbo-0613": 4096,
#     "gpt-3.5-turbo-16k-0613": 16384,
#     "text-curie-001": 2049,
#     "text-babbage-001": 2049,
#     "text-ada-001": 2049,
#     "davinci": 2049,
#     "curie": 2049,
#     "babbage": 2049,
#     "ada": 2049
# }
CHATGPT_MODEL = "gpt-3.5-turbo-16k-0613"
MODEL_MAX_TOKENS = 16384


def call_openai_api(prompt, transcription):
    """
    Calls the OpenAI API with the specified prompt and transcription.
    """
    try:
        response = openai.ChatCompletion.create(
            model=CHATGPT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcription},
            ],
        )
        return response['choices'][0]['message']['content']
    except OpenAIError as e:
        print(f"OpenAI API call failed: {e}")
        raise


@retry(retry=retry_if_exception_type(openai.error.RateLimitError), wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(7))
def ask_chatgpt_to_summarize(transcription):
    """
    Asks ChatGPT to summarize a transcription.
    """
    print("Summarizing transcription due to length.")
    prompt = "You are a highly skilled AI trained in language comprehension and summarization. I would " \
             "like you to read the following text and summarize it into concise abstract paragraphs. " \
             "Strategies for summarization: " \
             "- Remove unnecessary words, phrases, or redundancies from the messages to make them " \
             "concise." \
             "- Remove unnecessary tokens such as HTML tags, URLs, and long numbers if they don't " \
             "impact the context." \
             " - Simplify language for a more concise summary."
    return call_openai_api(prompt, transcription)


def shrink_transcription(transcription):
    """
    Shrinks a transcription if it exceeds the maximum token count.
    """
    encoding = tiktoken.encoding_for_model(CHATGPT_MODEL)
    num_tokens = len(encoding.encode(transcription))
    print(f"Transcription has {num_tokens} tokens.")
    max_tokens = int(MODEL_MAX_TOKENS * .75)
    print(f"Max tokens: {max_tokens} for model {CHATGPT_MODEL}")

    if num_tokens <= max_tokens:
        print("Transcription is short enough. No need to summarize.")
        return transcription

    sentences = collections.deque(re.split(r'(?<=[.!?]) +', transcription))
    buffer = ""

    while sentences:
        sentence = sentences.popleft()
        temp_buffer = buffer + " " + sentence

        tokens = len(encoding.encode(temp_buffer))
        print(f"Buffer + sentence has {tokens} tokens.")

        if tokens > max_tokens:
            buffer = ask_chatgpt_to_summarize(buffer)

            tokens = len(encoding.encode(buffer + " " + sentence))
            print(f"After summarization, buffer + sentence has {tokens} tokens.")
            if tokens > max_tokens:
                raise Exception(
                    f"Transcription is too long to summarize. "
                    f"Current summarized buffer + sentence = {tokens} tokens."
                )

            sentences.appendleft(sentence)
        else:
            buffer = temp_buffer

    print(f"Transcription shrunk, length: {len(buffer)}.")
    return buffer


def generate_response(txt):
    # Instantiate the LLM model
    # llm = ChatOpenAI(temperature=0, model_name=CHATGPT_MODEL)

    shrunk_text = shrink_transcription(txt)
    print(shrunk_text)

    # # Split text
    # text_splitter = TokenTextSplitter(chunk_size=8192, chunk_overlap=800)
    # texts = text_splitter.split_text(txt)
    # Create multiple documents
    # chain_type = 'stuff'
    # chain = load_summarize_chain(llm, chain_type=chain_type)
    # return chain.run([Document(page_content=shrunk_text)])

    # prompt = "You are a highly skilled AI trained in language comprehension and summarization. I would " \
    #          "like you to read the following text and summarize it into concise abstract paragraphs. " \
    #          "Aim to retain the most important points, providing a coherent and readable summary that " \
    #          "could help a person understand the main points of the discussion without needing to read " \
    #          "the entire text. Use details from the text to support your summary, but do not include " \
    #          "any information that is not in the text. Be as detailed as possible while remaining " \
    #          "concise."

    # prompt = ("Provide a brief, coherent summary of the text's main points, arguments, and supporting details. Focus "
    #           "on the central thesis and key evidence, including relevant data, examples, and any opposing "
    #           "viewpoints. Describe the text's structure and style briefly, and conclude by highlighting its "
    #           "significance or impact within its field, maintaining clarity and conciseness.")

    prompt = """
    Objective: Provide a coherent and detailed summary of the key points, arguments, and important aspects of the given text while ensuring readability.

    Key Points and Arguments:

        Summarize the main thesis or central argument of the text.
        List the most significant supporting arguments or evidence presented.
        Identify any counterarguments or opposing viewpoints discussed.

    Important Aspects and Details:
    4. Highlight any specific data, statistics, or examples used to support the author's claims.

        Mention notable case studies, anecdotes, or real-world applications if applicable.
        Identify key terminology, concepts, or theories introduced and explain them briefly.
        Note any historical or contextual information that adds depth to the text.
        Highlight any relevant quotes or excerpts that encapsulate important ideas.

    Structure and Style:
    9. Describe the text's overall structure and organization (e.g., chronological, thematic, problem-solution).

        Comment on the author's writing style, tone, and any rhetorical devices employed (e.g., anecdotes, metaphors).

    Conclusion:
    11. Summarize the overall significance or impact of the text's content.

        Provide insights into how the text contributes to the broader discourse or field (if applicable).

    Note: Please ensure that your summary is coherent and readable, maintaining the logical flow of the original text. Include relevant details from the text to support your summary while avoiding excessive repetition or extraneous information.
    """

    # horrifying prompt. don't use this near children.
    # prompt = """Summarize this text and emphasize the scary parts, making it sound like a horror story"""
    return call_openai_api(prompt, shrunk_text)


def list_gpt_model_ids():
    models = openai.Model.list()['data']
    gpt_model_ids = [model['id'] for model in models if model['id'].startswith('gpt-')]
    return gpt_model_ids


if __name__ == '__main__':
    print("Starting up...")

    # Page title
    st.set_page_config(page_title='🦜🔗 Text Summarization App')
    st.title('🦜🔗 Text Summarization App')

    # Text input
    txt_input = st.text_area('Enter your text', '', height=200)
    # get all model ids that are available to the user (based on their API key)
    visible_model_ids = list_gpt_model_ids()

    # Form to accept user's text input for summarization
    result = []
    with st.form('summarize_form', clear_on_submit=True):
        # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not txt_input)
        submitted = st.form_submit_button('Submit')
        if submitted:
            with st.spinner('Calculating...'):
                response = generate_response(txt_input)
                result.append(response)

    if len(result):
        st.info(response)
