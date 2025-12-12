import os
import logging
from dataclasses import dataclass
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ローカル開発用（Cloudでは .env が無くてもOK）
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# 定数とデータクラス
# =========================
@dataclass
class ExpertConfig:
    """専門家タイプの設定"""
    name: str
    system_message: str


class AppConstants:
    """アプリケーション定数"""
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.3
    PAGE_TITLE = "LangChain LLM App"
    PAGE_ICON = "🤖"

    # 専門家タイプの定義
    EXPERT_CONFIGS = {
        "A：キャリア相談のプロ（転職・職務経歴書・面接）": ExpertConfig(
            name="A：キャリア相談のプロ（転職・職務経歴書・面接）",
            system_message=(
                "あなたはキャリア相談のプロです。ユーザーの状況を整理し、"
                "現実的で具体的な次の一手を提案してください。必要なら質問もしてください。"
            ),
        ),
        "B：Python/生成AIの講師（初心者向け）": ExpertConfig(
            name="B：Python/生成AIの講師（初心者向け）",
            system_message=(
                "あなたはPythonと生成AIの初心者向け講師です。"
                "専門用語はかみ砕き、手順を番号付きで具体的に説明してください。"
                "可能ならコピペできる例も示してください。"
            ),
        ),
    }

    DEFAULT_SYSTEM_MESSAGE = "あなたは親切で有能なアシスタントです。"


# =========================
# キー取得（Cloud対応の肝）
# =========================
def get_api_key() -> str:
    """
    Streamlit Community Cloud: st.secrets から取得
    ローカル: 環境変数 or .env（load_dotenv済み）から取得
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise ValueError("OPENAI_API_KEY が設定されていません")
    return api_key


# =========================
# ヘルパー関数
# =========================
def get_system_message(expert_type: str) -> str:
    """専門家タイプに基づいてシステムメッセージを取得"""
    config = AppConstants.EXPERT_CONFIGS.get(expert_type)
    if config:
        return config.system_message
    logger.warning(f"Unknown expert type: {expert_type}. Using default message.")
    return AppConstants.DEFAULT_SYSTEM_MESSAGE


@st.cache_resource
def get_llm(api_key: str) -> ChatOpenAI:
    """LLMインスタンスをキャッシュして再利用（安定＆高速化）"""
    return ChatOpenAI(
        model=AppConstants.MODEL_NAME,
        temperature=AppConstants.TEMPERATURE,
        api_key=api_key,
    )


# =========================
# LLM呼び出し関数（条件：引数2つ→戻り値1つ）
# =========================
def ask_llm(input_text: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、LLMの回答を返す
    """
    logger.info(f"LLM呼び出し開始 - Expert: {expert_type}")

    system_message = get_system_message(expert_type)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}"),
        ]
    )

    api_key = get_api_key()
    llm = get_llm(api_key)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": input_text})

    logger.info("LLM呼び出し成功")
    return response


# =========================
# セッション状態管理
# =========================
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0


def add_to_history(user_input: str, expert_type: str, response: str):
    st.session_state.chat_history.append(
        {
            "user_input": user_input,
            "expert_type": expert_type,
            "response": response,
            "timestamp": st.session_state.message_count,
        }
    )
    st.session_state.message_count += 1


def display_chat_history():
    if st.session_state.chat_history:
        st.subheader("📝 会話履歴")
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(
                f"会話 {len(st.session_state.chat_history) - i + 1}: {chat['expert_type'][:20]}..."
            ):
                st.markdown(f"**質問:**\n{chat['user_input']}")
                st.markdown(f"**回答:**\n{chat['response']}")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=AppConstants.PAGE_TITLE, page_icon=AppConstants.PAGE_ICON)
initialize_session_state()

st.title("🤖 LangChain × Streamlit LLMアプリ")

st.markdown(
    """
### このアプリでできること
- 入力フォームにテキストを入力して送信すると、**LangChain経由でLLMに問い合わせ**て回答を表示します。
- ラジオボタンで「LLMに振る舞わせる専門家」を選べます（**選択に応じてシステムメッセージが切り替わります**）。
- 会話履歴が保存され、サイドバーで確認できます。

### 使い方
1. 「専門家タイプ」を選ぶ（A or B）
2. 下の入力欄に質問を入力
3. 「送信」を押す
"""
)

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")

    if st.button("🗑️ 会話履歴をクリア"):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.success("会話履歴をクリアしました")
        st.rerun()

    st.metric("会話数", len(st.session_state.chat_history))
    st.divider()
    display_chat_history()


# APIキー存在チェック（Cloud/ローカル両対応）
try:
    _ = get_api_key()
except ValueError:
    st.error(
        "OPENAI_API_KEY が見つかりません。\n\n"
        "- ローカル: `.env` に `OPENAI_API_KEY=...` を設定\n"
        "- Streamlit Community Cloud: Secrets に `OPENAI_API_KEY` を設定"
    )
    st.stop()


expert_type = st.radio(
    "専門家タイプを選択してください",
    [
        "A：キャリア相談のプロ（転職・職務経歴書・面接）",
        "B：Python/生成AIの講師（初心者向け）",
    ],
)

user_text = st.text_area(
    "入力フォーム（質問・依頼内容）",
    placeholder="例：職務経歴書の要約文を改善したい / LangChainの基本を手順で教えて",
    height=140,
    key="user_text",
)

# 送信/クリア
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit_button = st.button("📤 送信", type="primary")
with col2:
    clear_button = st.button("🔄 クリア")

if clear_button:
    st.session_state["user_text"] = ""
    st.rerun()

if submit_button:
    if not user_text.strip():
        st.warning("⚠️ 入力フォームにテキストを入力してください。")
        st.stop()

    if len(user_text) > 2000:
        st.warning("⚠️ 入力が長すぎます。2000文字以内にしてください。")
        st.stop()

    with st.spinner("🤔 LLMに問い合わせ中..."):
        try:
            answer = ask_llm(user_text, expert_type)
            add_to_history(user_text, expert_type, answer)
            st.success("✅ 回答を取得しました！")
        except Exception as e:
            st.error("❌ LLM呼び出しでエラーが発生しました。")
            logger.exception("Unexpected error during LLM call")
            st.exception(e)
            st.stop()

    st.subheader("💬 回答")
    st.markdown(answer)

    st.divider()
    col_fb1, col_fb2, col_fb3 = st.columns(3)
    with col_fb1:
        if st.button("👍 役立った"):
            st.success("フィードバックありがとうございます！")
    with col_fb2:
        if st.button("👎 改善が必要"):
            st.info("フィードバックありがとうございます。改善に努めます！")
