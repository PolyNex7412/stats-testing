"""AI-powered analysis assistant using Claude API (streaming)."""
import os
import numpy as np
import anthropic
from typing import Generator

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """あなたは統計分析の専門家アシスタントです。
データ分析に不慣れなユーザーが統計的仮説検定の結果を正しく理解できるよう、
わかりやすい日本語で解説・示唆を提供してください。

## スタイルガイド
- 専門用語には必ず平易な補足説明を添える
- 絵文字を適度に使い、親しみやすいトーンにする
- 箇条書き・見出しで読みやすく整形する
- 数式は最小限に抑え、概念的な理解を優先する
- ビジネス・研究における具体的な含意を必ず述べる
- 「次にすべきこと」を必ず提案する
- 不確実な点は正直に伝え、過度な断言を避ける"""


# ── Context builder ───────────────────────────────────────────────────────────
def build_test_context(result) -> str:
    """Serialize test result into a readable context string."""
    lines = [
        f"検定手法: {result.test_name}",
        f"帰無仮説 (H₀): {result.h0}",
        f"対立仮説 (H₁): {result.h1}",
        f"有意水準 (α): {result.alpha}",
        f"検定統計量: {result.statistic:.4f}",
        f"p値: {result.p_value:.4f}",
        f"判定: {'🟢 帰無仮説を棄却（統計的に有意）' if result.reject_h0 else '🔴 帰無仮説を棄却できない（統計的に有意でない）'}",
    ]
    if result.df is not None:
        lines.append(f"自由度 (df): {result.df:.1f}")
    if result.ci is not None:
        lines.append(f"{int((1-result.alpha)*100)}% 信頼区間: [{result.ci[0]:.4f}, {result.ci[1]:.4f}]")
    if result.effect_size is not None:
        lines.append(f"効果量 (Cohen's d): {result.effect_size:.3f}")
    if result.n is not None:
        lines.append(f"サンプルサイズ n₁: {result.n}")
    if result.n2 is not None:
        lines.append(f"サンプルサイズ n₂: {result.n2}")
    if result.mean1 is not None:
        lines.append(f"平均1: {result.mean1:.4f}")
    if result.mean2 is not None:
        lines.append(f"平均2: {result.mean2:.4f}")
    if result.std1 is not None:
        lines.append(f"標準偏差1: {result.std1:.4f}")
    return "\n".join(lines)


def build_data_summary(data_dict: dict) -> str:
    if not data_dict:
        return ""
    lines = []
    for label, data in data_dict.items():
        arr = np.asarray(data)
        lines.append(
            f"{label}: n={len(arr)}, 平均={arr.mean():.3f}, "
            f"SD={arr.std(ddof=1):.3f}, 中央値={np.median(arr):.3f}, "
            f"最小={arr.min():.3f}, 最大={arr.max():.3f}"
        )
    return "\n".join(lines)


# ── Streaming generators ──────────────────────────────────────────────────────
def auto_insight_stream(result, data_dict: dict) -> Generator[str, None, None]:
    """Yield streaming tokens for the automatic insight report."""
    client = anthropic.Anthropic()

    ctx = build_test_context(result)
    data_summary = build_data_summary(data_dict)

    user_msg = f"""以下の統計検定の結果を分析し、データ分析初心者にもわかりやすく示唆を提供してください。

【検定結果】
{ctx}
{f"【データの概要】{chr(10)}{data_summary}" if data_summary else ""}

以下の構成で、合計400〜600字程度でまとめてください:

## 🎯 ひとことまとめ
（3行以内で核心を突いたまとめ）

## 📊 詳しい解釈
（結果が実際に何を意味するのかを具体的に）

## ⚠️ 注意点
（解釈の限界・サンプルサイズ・前提条件など）

## 🔍 次のステップ
（追加で行うべき分析や確認事項の提案）"""

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def chat_stream(
    chat_history: list[dict],
    result,
    data_dict: dict,
) -> Generator[str, None, None]:
    """Yield streaming tokens for a follow-up question."""
    client = anthropic.Anthropic()

    ctx = build_test_context(result)
    data_summary = build_data_summary(data_dict)

    system = f"""{SYSTEM_PROMPT}

---
## 現在の検定結果（参照用）
{ctx}
{f"{chr(10)}## データの概要{chr(10)}{data_summary}" if data_summary else ""}
---

ユーザーが深堀り質問をしてきます。上記の検定結果を踏まえて、簡潔かつ丁寧に日本語で回答してください。"""

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=800,
        system=system,
        messages=chat_history,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── Preset questions ──────────────────────────────────────────────────────────
def get_preset_questions(result) -> list[str]:
    """Return context-aware preset questions for beginners."""
    base = [
        "この結果をビジネスの意思決定にどう活かせますか？",
        "サンプルサイズは十分でしたか？もっと必要でしたか？",
        "次にどんな分析をすると、より深い洞察が得られますか？",
        "p値が有意水準ギリギリの場合、どう判断すればよいですか？",
    ]

    if result.effect_size is not None:
        if abs(result.effect_size) < 0.2:
            base.insert(0, "効果量が小さいですが、それでも実用的に意味がありますか？")
        elif abs(result.effect_size) > 0.8:
            base.insert(0, "効果量が大きいとはどういう意味ですか？実務での含意は？")

    if not result.reject_h0:
        base.insert(0, "有意差がなかったのは、本当に差がないからですか？それとも別の理由？")
    else:
        base.insert(0, "統計的に有意というだけで、実際に重要な差があると言えますか？")

    return base[:5]


def check_api_key() -> bool:
    """Return True if ANTHROPIC_API_KEY is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
