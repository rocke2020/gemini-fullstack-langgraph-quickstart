import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from openai import OpenAI
from pydantic import SecretStr

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-xxx")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-xxx")
DEEPSEEK32_MAX_TOKENS = 131072  # 128k
DEEPSEEK_CHAT_MODEL = "deepseek-chat"


def get_model_client(
    model_name="glm-4.7",
    api_type="tongyi",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
):
    if api_type == "tongyi":
        return ChatTongyi(
            model=model_name,
            api_key=SecretStr(DASHSCOPE_API_KEY),
            max_retries=max_retries,
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout if timeout else 60,
            },
        )
    elif api_type == "deepseek":
        return ChatDeepSeek(
            model=DEEPSEEK_CHAT_MODEL,
            max_retries=max_retries,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


deepseek_model = ChatDeepSeek(
    model=DEEPSEEK_CHAT_MODEL,
    temperature=0,
    max_tokens=None,
    max_retries=2,
    timeout=None,
    # other params...
)

# free 90days:  glm-4.7：200k, deepseek-v3.2: 128k, qwen3-vl-plus-2025-12-19 qwen-plus-2025-12-01
# paid model: qwen3-plus
tongyi_chat_model = ChatTongyi(
    model="glm-4.7",
    api_key=SecretStr(DASHSCOPE_API_KEY),
    max_retries=2,
    model_kwargs={"temperature": 0.7},
)
tongyi_vl_plus_model = ChatTongyi(
    model="qwen3-vl-plus-2025-12-19",
    api_key=SecretStr(DASHSCOPE_API_KEY),
    max_retries=2,
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",  # or "text-embedding-v1"
    dashscope_api_key=DASHSCOPE_API_KEY,  # optional if set in env
)

messages_raw = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
messages_formatted = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Chinese."
    ),
    HumanMessage(
        content="Translate this sentence from English to Chinese. I love programming."
    ),
]

deepseek_openai_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)
tongyi_openai_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages_dict = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]


def test_openai_client(client, model, messages: list[dict]):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    print(response.model_dump_json())
    print(response.choices[0].message.content)


def test_openai_client_deepseek(model_name="deepseek-chat"):
    test_openai_client(deepseek_openai_client, model_name, messages_dict)


def test_openai_client_tongyi(model_name="qwen3-vl-plus-2025-12-19"):
    test_openai_client(tongyi_openai_client, model_name, messages_dict)


if __name__ == "__main__":
    RUN_DEEPSEEK = 0
    RUN_QW_CHAT = 0
    RUN_QW_OPENAI = 0
    if RUN_DEEPSEEK:
        ai_msg = deepseek_model.invoke(messages_raw)
        """ The returned ai_msg is of type HumanMessage
ai_msg = AIMessage(
    content='我喜欢编程。', additional_kwargs={},
    response_metadata= {
        'model_name': 'deepseek-v3.2',
        'finish_reason': 'stop',
        'request_id': '12f5c94c-8f34-40c2-b36f-4357cd44a22c',
        'token_usage': {'input_tokens': 24, 'output_tokens': 3, 'total_tokens': 27}},
    id='lc_run--019b842c-921d-7fe1-ab98-a08c0406dc9b-0'
)
        """
        print(f"deepseek: {ai_msg = }")

    if RUN_QW_CHAT:
        ai_msg = tongyi_chat_model.invoke(messages_raw)
        print(f"tongyi: {ai_msg = }")
        # ai_msg = tongyi_chat_model.invoke(messages_raw)
        # print(f"tongyi: {ai_msg = }")

    if RUN_QW_OPENAI:
        test_openai_client_tongyi()
