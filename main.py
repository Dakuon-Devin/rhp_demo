import os
import random
import openai
import google.generativeai as genai
import matplotlib.pyplot as plt

# ==========================
# 環境変数の設定（APIキー）
# ==========================
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================
# ルールの設定
# ==========================
RULES = [
    "文章は簡潔かつ明瞭であること。",
    "技術的な用語の誤用がないこと。",
    "論理的な構成になっていること。",
    "指定された要件を満たしていること。",
    "誤字・脱字がないこと。"
]

# ==========================
# 初期プロンプト
# ==========================
INITIAL_PROMPT = "与えられたテーマについて詳細な説明を行い、専門的かつ明確な文章を作成してください。: 強化学習"

# ==========================
# Gemini APIでテキスト生成
# ==========================
def generate_text_with_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text if response else ""

# ==========================
# GPT-4 APIで報酬計算
# ==========================
def evaluate_text(text: str) -> float:
    """
    GPT-4で生成された文章を評価し、スコアを返す（0.0 - 1.0）。
    """
    gpt_eval_prompt = f"""
    以下の文章を評価してください。
    - 文章が明確で論理的であるか？
    - 指定された要件を満たしているか？
    - 誤字脱字がないか？
    文章:
    """{text}"""
    回答は0.0〜1.0のスコアのみで出力してください。
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": gpt_eval_prompt}]
        )
        gpt_score = float(response["choices"][0]["message"]["content"].strip())
        return min(1.0, max(0.0, gpt_score))
    except Exception as e:
        print("GPT-4評価エラー:", e)
        return 0.0

# ==========================
# Gemini APIでプロンプトの改良（GAの変異操作）
# ==========================
def mutate_prompt(prompt: str) -> str:
    """
    プロンプトをランダムに変異させる（ランダムな単語の追加・変更）。
    """
    mutations = [
        "より詳しく説明してください。",
        "簡潔にまとめてください。",
        "実例を加えてください。",
        "技術的な詳細を含めてください。",
        "論理的な順序で整理してください。"
    ]
    mutation = random.choice(mutations)
    return f"{prompt} {mutation}"

# ==========================
# GAでプロンプト最適化（スコアの可視化付き）
# ==========================
def train_prompt_optimization():
    """GAを使ってプロンプトを進化させ、スコアを可視化する"""
    population = [(INITIAL_PROMPT, 0.5)]  # 初期スコアを適当に設定
    scores = []
    
    for step in range(100):  # 100ステップ学習
        prompt, prev_score = random.choice(population)
        mutated_prompt = mutate_prompt(prompt)  # プロンプトをランダム変異
        generated_text = generate_text_with_gemini(mutated_prompt)
        reward = evaluate_text(generated_text)
        scores.append(reward)
        
        # スコアが高い場合、新しいプロンプトをリストに追加
        if reward > 0.5:
            population.append((mutated_prompt, reward))
        
        # 世代交代：低スコアのプロンプトを削除
        if len(population) > 10:
            population = sorted(population, key=lambda x: x[1], reverse=True)[:10]
        
        print(f"Step {step}: Reward={reward:.4f}, New Prompt={mutated_prompt}")
    
    # スコアの推移を可視化
    plt.plot(scores)
    plt.xlabel("Step")
    plt.ylabel("Evaluation Score")
    plt.title("Evolution of Prompt Scores")
    plt.show()

if __name__ == "__main__":
    train_prompt_optimization()
