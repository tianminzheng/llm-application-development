
import base64
import json
import os
import re
import requests
import time
from io import BytesIO
from typing import Union, List, Literal, Optional, Dict, Any

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, UnidentifiedImageError
from audio_recorder_streamlit import audio_recorder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from matplotlib.figure import Figure
from openai import OpenAI
from openai._legacy_response import HttpxBinaryResponseContent


def initialize_session_state_variables() -> None:
    """
    初始化session state变量
    """

    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "openai" not in st.session_state:
        st.session_state.openai = None

    if "history" not in st.session_state:
        st.session_state.history = []

    if "model_type" not in st.session_state:
        st.session_state.model_type = "GPT Models from OpenAI"

    if "ai_role" not in st.session_state:
        st.session_state.ai_role = 2 * ["你是一个有用的人工智能助手。"]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "temperature" not in st.session_state:
        st.session_state.temperature = [0.7, 0.7]

    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "mic_used" not in st.session_state:
        st.session_state.mic_used = False

    if "audio_response" not in st.session_state:
        st.session_state.audio_response = None

    if "image_url" not in st.session_state:
        st.session_state.image_url = None

    if "image_description" not in st.session_state:
        st.session_state.image_description = None

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "google_cse_id_validity" not in st.session_state:
        st.session_state.google_cse_id_validity = False

    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: Any, **kwargs) -> None:
        new_text = self._extract_text(token)
        if new_text:
            self.text += new_text
            self.container.markdown(self.text)

    def _extract_text(self, token: Any) -> str:
        if isinstance(token, str):
            return token
        elif isinstance(token, list):
            return ''.join(self._extract_text(t) for t in token)
        elif isinstance(token, dict):
            return token.get('text', '')
        else:
            return str(token)


def is_openai_api_key_valid(openai_api_key: str) -> bool:
    """
    如果给定的OpenAI API密钥有效，则返回True
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_anthropic_api_key_valid(anthropic_api_key: str) -> bool:
    """
    如果给定的Anthropic API密钥有效，则返回True。
    """

    headers = {
        "x-api-key": anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-2.1",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_google_api_key_valid(google_api_key: str) -> bool:
    """
    如果给定的Google API密钥有效，则返回True
    """

    if not google_api_key:
        return False

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=google_api_key
    )
    try:
        gemini_llm.invoke("Hello")
    except:
        return False
    else:
        return True


def are_google_api_key_cse_id_valid(
    google_api_key: str, google_cse_id: str
) -> bool:

    """
    如果提供的Google API密钥和CSE ID有效，则返回True。
    """

    if google_api_key and google_cse_id:
        try:
            search = GoogleSearchAPIWrapper(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id,
                k=1
            )
            search.run("我在哪里可以获得一个Google CSE ID？")
        except:
            return False
        else:
            return True
    else:
        return False


def check_api_keys() -> None:
    # 取消设置这个标志以检查API密钥的有效性
    st.session_state.ready = False


def message_history_to_string(extra_space: bool=True) -> str:
    """
    返回包含在st.session_state.history中的聊天记录的字符串
    """

    history_list = []
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            history_list.append(f"Human: {msg.content}")
        else:
            history_list.append(f"AI: {msg.content}")
    new_lines = "\n\n" if extra_space else "\n"

    return new_lines.join(history_list)


def get_chat_model(
    model: str,
    temperature: float,
    callbacks: List[BaseCallbackHandler]
) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, None]:

    """
    根据给定的模型名称获取相应的聊天模型。
    """

    model_map = {
        "gpt-": ChatOpenAI,
        "claude-": ChatAnthropic,
        "gemini-": ChatGoogleGenerativeAI
    }
    for prefix, ModelClass in model_map.items():
        if model.startswith(prefix):
            return ModelClass(
                model=model,
                temperature=temperature,
                streaming=True,
                callbacks=callbacks
            )
    return None


def process_with_images(
    llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI],
    message_content: str,
    image_urls: List[str]
) -> str:

    """
    使用语言模型处理给定的历史查询及其相关联的图像。
    """

    print(message_content)

    content_with_images = (
        [{"type": "text", "text": message_content}] +
        [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    )
    message_with_images = [HumanMessage(content=content_with_images)]

    return llm.invoke(message_with_images).content


def perform_query(
    query: str,
    model: str,
    image_urls: List[str],
    temperature: float=0.7,
) -> Union[str, None]:

    """
    根据用户查询生成文本。
    聊天提示和消息历史存储在st.session_state变量中。
    """

    try:
        llm = get_chat_model(model, temperature, [StreamHandler(st.empty())])
        if llm is None:
            st.error(f"不支持的模型: {model}", icon="🚨")
            return None

        # 获取聊天历史
        chat_history = st.session_state.history
        print(chat_history)

        history_query = {"chat_history": chat_history, "input": query}
        print(history_query)

        # 获取系统消息
        message_with_no_image = st.session_state.chat_prompt.invoke(history_query)
        print(message_with_no_image)
        message_content = message_with_no_image.messages[0].content
        print(message_content)

        # 执行图片对话
        if image_urls:
            generated_text = process_with_images(llm, message_content, image_urls)
            human_message = HumanMessage(
                content=query, additional_kwargs={"image_urls": image_urls}
            )
        # 执行文本对话
        else:
            generated_text = llm.invoke(message_with_no_image).content
            human_message = HumanMessage(content=query)

        if isinstance(generated_text, list):
            generated_text = generated_text[0]["text"]

        # 添加聊天历史
        st.session_state.history.append(human_message)
        st.session_state.history.append(AIMessage(content=generated_text))
        print(st.session_state.history)

        return generated_text
    except Exception as e:
        st.error(f"出现异常: {e}", icon="🚨")
        return None


def openai_create_image(
    description: str, model: str="dall-e-3", size: str="1024x1024"
) -> Optional[str]:

    """基于描述生成图像"""

    try:
        with st.spinner("AI is generating..."):
            response = st.session_state.openai.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
    except Exception as e:
        image_url = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_url

def display_text_with_equations(text: str):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def read_audio(audio_bytes: bytes) -> Optional[str]:
    """
    读取音频流并返回对应的文本
    """
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.openai.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"出现异常: {e}", icon="🚨")

    return text


def input_from_mic() -> Optional[str]:
    """
    将麦克风的音频输入转换为文本并返回。
    如果没有音频输入，则返回None。
    """

    time.sleep(0.5)
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def perform_tts(text: str) -> Optional[HttpxBinaryResponseContent]:
    """
    将文本作为输入，执行文本转语音（TTS），并返回一个音频响应。
    """

    try:
        with st.spinner("TTS处理中..."):
            audio_response = st.session_state.openai.audio.speech.create(
                model="tts-1",
                voice="fable",
                input=text,
            )
    except Exception as e:
        audio_response = None
        st.error(f"发生异常: {e}", icon="🚨")

    return audio_response


def play_audio(audio_response: HttpxBinaryResponseContent) -> None:
    """
    将文本转语音（TTS）生成的音频响应作为输入，并播放该音频。
    """

    audio_data = audio_response.read()

    # 将音频数据编码为Base64格式
    b64 = base64.b64encode(audio_data).decode("utf-8")

    # 创建一个Markdown字符串，用于嵌入带有Base64源的音频播放器。
    md = f"""
        <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        你的浏览器不支持语音元素.
        </audio>
        """

    # 使用Streamlit来渲染音频播放器
    st.markdown(md, unsafe_allow_html=True)


def image_to_base64(image: Image) -> str:
    """
    将PIL图像对象转换为Base64编码的图像，并返回得到的编码图像字符串，以便用作URL的替代品。
    """

    # 将当前图像转换为RGB模式，并且返回新的图像
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 将图像保存为BytesIO对象
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # 将BytesIO对象转换为字节并采用base64进行编码
    img_str = base64.b64encode(buffered_image.getvalue())

    # 将字节转换为字符串
    base64_image = img_str.decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def shorten_image(image: Image, max_pixels: int=1024) -> Image:
    """
    接收一个图像对象作为输入，如果图像大小超过最大像素 x 最大像素的限制，则缩短图像尺寸。
    """

    if max(image.width, image.height) > max_pixels:
        if image.width > image.height:
            new_width, new_height = 1024, image.height * 1024 // image.width
        else:
            new_width, new_height = image.width * 1024 // image.height, 1024

        image = image.resize((new_width, new_height))

    return image


def upload_image_files_return_urls(
    type: List[str]=["jpg", "jpeg", "png", "bmp"]
) -> List[str]:

    """
    上传图像文件，将它们转换为Base64编码的图像，并返回结果编码图像的列表。
    """

    st.write("")
    st.write("**图像对话**")
    source = st.radio(
        label="图像选择",
        options=("Uploaded", "From URL"),
        horizontal=True,
        label_visibility="collapsed",
    )
    image_urls = []

    if source == "Uploaded":
        uploaded_files = st.file_uploader(
            label="上传图像",
            type=type,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="image_upload_" + str(st.session_state.uploader_key),
        )
        if uploaded_files:
            try:
                for image_file in uploaded_files:
                    image = Image.open(image_file)
                    thumbnail = shorten_image(image, 300)
                    st.image(thumbnail)
                    image = shorten_image(image, 1024)
                    image_urls.append(image_to_base64(image))
            except UnidentifiedImageError as e:
                st.error(f"发生错误: {e}", icon="🚨")
    else:
        image_url = st.text_input(
            label="图像URL",
            label_visibility="collapsed",
            key="image_url_" + str(st.session_state.uploader_key),
        )
        if image_url:
            if is_url(image_url):
                st.image(image_url)
                image_urls = [image_url]
            else:
                st.error("输入一个正确的URL", icon="🚨")

    return image_urls


def fig_to_base64(fig: Figure) -> str:
    """
    将一个图形对象转换为Base64编码的图像，并返回得到的编码图像，以便代替URL使用。
    """

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image = Image.open(buffer)

    return image_to_base64(image)


def is_url(text: str) -> bool:
    """
    确实文本是否为一个合法的URL
    """

    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False


def reset_conversation() -> None:
    """
    重置session_state变量以重置对话
    """

    st.session_state.history = []
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.uploader_key = 0


def switch_between_apps() -> None:
    """
    Keep the chat settings when switching the mode.
    """

    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]


def set_prompts() -> None:
    """
    设置提示词
    """

    st.session_state.chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{st.session_state.ai_role[0]} "
            f"你的目标是回答人类的询问。如果信息不可用，明确告知人类无法找到答案。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])


def print_conversation(no_of_msgs: Union[Literal["All"], int]) -> None:
    """
    打印存储在st.session_state.history中的对话
    """

    if no_of_msgs == "All":
        no_of_msgs = len(st.session_state.history)

    for msg in st.session_state.history[-no_of_msgs:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.write(msg.content)
        else:
            with st.chat_message("ai"):
                display_text_with_equations(msg.content)

        if urls := msg.additional_kwargs.get("image_urls"):
            for url in urls:
                st.image(url)

    # 执行TTS
    if (
        st.session_state.model_type == "GPT Models from OpenAI"
        and st.session_state.audio_response is not None
    ):
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None


def deserialize_messages(
    serialized_messages: List[Dict]
) -> List[Union[HumanMessage, AIMessage]]:

    """
    从字典列表反序列化消息列表
    """

    deserialized_messages = []
    for msg in serialized_messages:
        if msg['type'] == 'human':
            deserialized_messages.append(HumanMessage(**msg))
        elif msg['type'] == 'ai':
            deserialized_messages.append(AIMessage(**msg))
    return deserialized_messages


def show_uploader() -> None:
    """
    设置显示上传器的标志
    """

    st.session_state.show_uploader = True


def check_conversation_keys(lst: List[Dict[str, Any]]) -> bool:
    """
    检查给定列表中的所有项目是否为有效的对话条目。
    """

    return all(
        isinstance(item, dict) and
        isinstance(item.get("content"), str) and
        isinstance(item.get("type"), str) and
        isinstance(item.get("additional_kwargs"), dict)
        for item in lst
    )


def load_conversation() -> bool:
    """
    从JSON文件加载对话
    """

    st.write("")
    st.write("**Choose a (JSON) conversation file**")
    uploaded_file = st.file_uploader(
        label="Load conversation", type="json", label_visibility="collapsed"
    )
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            if isinstance(data, list) and check_conversation_keys(data):
                st.session_state.history = deserialize_messages(data)
                return True
            st.error(
                f"The uploaded data does not conform to the expected format.", icon="🚨"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")

    return False


def create_text(model: str) -> None:
    """
    以LLM作为输入，并用户输入生成文本
    """

    general_role = "你是一个有用的人工智能助手。"
    roles = (
        general_role
    )

    with st.sidebar:

        if st.session_state.model_type == "GPT Models from OpenAI":
            st.write("")
            st.write("**TTS**")
            st.session_state.tts = st.radio(
                label="TTS",
                options=("Enabled", "Disabled", "Auto"),
                # horizontal=True,
                index=1,
                label_visibility="collapsed",
            )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature[0] = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature[1],
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("")
        st.write("**显示消息**")
        no_of_msgs = st.radio(
            label="显示消息",
            options=("All", 20, 10),
            label_visibility="collapsed",
            horizontal=True,
            index=2,
        )

    st.write("")
    st.write("##### 发送AI消息")
    st.session_state.ai_role[0] = st.selectbox(
        label="AI的角色",
        options=roles,
        index=roles.index(st.session_state.ai_role[1]),
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        reset_conversation()
        st.rerun()

    st.write("")
    st.write("##### 和AI对话")

    # Print conversation
    print_conversation(no_of_msgs)

    if st.session_state.show_uploader and load_conversation():
        st.session_state.show_uploader = False
        st.rerun()

    set_prompts()

    with st.sidebar:
        image_urls = upload_image_files_return_urls()

    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            query = audio_input
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True

    text_input = st.chat_input(placeholder="输入查询条件")

    if text_input:
        query = text_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(query)

        with st.chat_message("ai"):
            generated_text = perform_query(
                query=query,
                model=model,
                image_urls=image_urls,
                temperature=st.session_state.temperature[0],
            )
            fig = plt.gcf()
            if fig and fig.get_axes():
                generated_image_url = fig_to_base64(fig)
                st.session_state.history[-1].additional_kwargs["image_urls"] = [
                    generated_image_url
                ]
        if (
            st.session_state.model_type == "GPT Models from OpenAI"
            and generated_text is not None
        ):
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)
            st.session_state.mic_used = False

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.session_state.uploader_key += 1
            st.rerun()


def create_image(model: str) -> None:
    """
    根据用户描述生成图像
    """

    with st.sidebar:
        st.write("")
        st.write("**像素**")
        image_size = st.radio(
            label="$\\hspace{0.1em}\\texttt{Pixel size}$",
            options=("1024x1024", "1792x1024", "1024x1792"),
            # horizontal=True,
            index=0,
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### 描述图像")

    if st.session_state.image_url is not None:
        st.info(st.session_state.image_description)
        st.image(image=st.session_state.image_url, use_column_width=True)

    # Get an image description using the microphone
    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            st.session_state.image_description = audio_input
            st.session_state.prompt_exists = True

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="输入图像的描述",
    )
    if text_input:
        st.session_state.image_description = text_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        st.session_state.image_url = openai_create_image(
            st.session_state.image_description, model, image_size
        )
        st.session_state.prompt_exists = False
        if st.session_state.image_url is not None:
            st.rerun()


def create_text_image() -> None:
    """
    使用LLM生成文本或图像
    """

    page_title = "LangChain多模态应用"
    page_icon = "📚"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="centered"
    )

    st.write(f"## {page_icon} $\,${page_title}")

    initialize_session_state_variables()

    with st.sidebar:

        st.write("")
        st.write("**Model Type**")
        st.session_state.model_type = st.sidebar.radio(
            label="Model type",
            options=(
                "GPT Models from OpenAI",
                "Claude Models from Anthropic",
                "Gemini Models from Google",
            ),
            on_change=check_api_keys,
            label_visibility="collapsed",
        )
        st.write("")
        if st.session_state.model_type in (
                "GPT Models from OpenAI", "Claude Models from Anthropic"
        ):
            validity = "(Verified)" if st.session_state.ready else ""
            if st.session_state.model_type == "GPT Models from OpenAI":
                st.write(
                    "**OpenAI API Key** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                openai_api_key = st.text_input(
                    label="OpenAI API Key",
                    type="password",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
            else:
                st.write(
                    "**Anthropic API Key** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                anthropic_api_key = st.text_input(
                    label="Anthropic API Key",
                    type="password",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
        else:
            validity = "(Verified)" if st.session_state.ready else ""
            st.write(
                "**Google API Key** ",
                f"<small>:blue[{validity}]</small>",
                unsafe_allow_html=True
            )
            google_api_key = st.text_input(
                label="Google API Key",
                type="password",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            if st.session_state.google_cse_id_validity:
                validity = "(Verified)"
            else:
                validity = ""
            st.write(
                "**Google CSE ID** ",
                f"<small>:blue[{validity}]</small>",
                unsafe_allow_html=True
            )
            google_cse_id = st.text_input(
                label="Google CSE ID",
                type="password",
                value="",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
        authentication = True

        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

    if authentication:
        if not st.session_state.ready:

            if st.session_state.model_type in (
                    "GPT Models from OpenAI", "Claude Models from Anthropic"
            ):
                if st.session_state.model_type == "GPT Models from OpenAI":
                    if is_openai_api_key_valid(openai_api_key):
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        st.session_state.openai = OpenAI()
                        st.session_state.ready = True
                else:
                    if is_anthropic_api_key_valid(anthropic_api_key):
                        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                        st.session_state.ready = True
            else:
                if is_google_api_key_valid(google_api_key):
                    os.environ["GOOGLE_API_KEY"] = google_api_key
                    st.session_state.ready = True
                    if are_google_api_key_cse_id_valid(
                            google_api_key, google_cse_id
                    ):
                        os.environ["GOOGLE_CSE_ID"] = google_cse_id
                        st.session_state.google_cse_id_validity = True
                    else:
                        st.session_state.google_cse_id_validity = False

            if st.session_state.ready:
                st.rerun()
            else:
                st.image("D:\\demo.png")
                st.stop()
    else:
        st.info("**在侧边栏输入正确的密码**")
        st.stop()

    gpt_models = ("gpt-4o-mini", "gpt-4o")
    claude_models = ("claude-3-haiku-20240307", "claude-3-5-sonnet-20240620")
    gemini_models = ("gemini-1.5-flash", "gemini-1.5-pro")

    with st.sidebar:
        st.write("")
        st.write("**Model**")

        if st.session_state.model_type == "GPT Models from OpenAI":
            model_options = gpt_models + ("dall-e-3",)
        elif st.session_state.model_type == "Claude Models from Anthropic":
            model_options = claude_models
        else:
            model_options = gemini_models

        model = st.radio(
            label="Models",
            options=model_options,
            label_visibility="collapsed",
            index=1,
            on_change=switch_between_apps,
        )

    if model == "dall-e-3":
        create_image(model)
    else:
        create_text(model)

if __name__ == "__main__":
    create_text_image()
