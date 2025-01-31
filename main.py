import os
import random
import openai
import google.generativeai as genai
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib for Japanese text
plt.rcParams['font.family'] = 'IPAGothic'
mpl.rcParams['figure.dpi'] = 300

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
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def generate_text_with_gemini(prompt: str) -> str:
    try:
        print(f"\nGenerating text with prompt: {prompt[:50]}...")
        model = genai.GenerativeModel("gemini-pro")
        with timeout(30):
            response = model.generate_content(prompt)
            result = response.text if response else ""
            print(f"Generated text length: {len(result)}")
            return result
    except TimeoutException:
        print("Gemini API call timed out after 30 seconds")
        return ""
    except Exception as e:
        print(f"Error in Gemini generation: {str(e)}")
        return ""

# ==========================
# GPT-4 APIで報酬計算
# ==========================
def evaluate_text(text: str) -> float:
    if not text:
        print("Empty text received, returning default score")
        return 0.5
        
    print("\nEvaluating text with GPT-4...")
    gpt_eval_prompt = f"""
    以下の文章を評価してください。
    - 文章が明確で論理的であるか？
    - 指定された要件を満たしているか？
    - 誤字脱字がないか？
    文章:
    {text}
    回答は0.0〜1.0のスコアのみで出力してください。
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": gpt_eval_prompt}]
        )
        print("GPT-4 response received")
        gpt_score = float(response["choices"][0]["message"]["content"].strip())
        result = min(1.0, max(0.0, gpt_score))
        print(f"Evaluation score: {result}")
        return result
    except Exception as e:
        print(f"GPT-4 evaluation error: {str(e)}")
        return 0.5

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
    try:
        print("Starting prompt optimization...")
        population = [(INITIAL_PROMPT, 0.5)]
        scores = []
        
        for step in range(100):  # 100ステップの最適化
            print(f"\nStarting step {step}")
            prompt, prev_score = random.choice(population)
            print("Selected prompt:", prompt[:50], "...")
            
            mutated_prompt = mutate_prompt(prompt)
            print("Mutated prompt:", mutated_prompt[:50], "...")
            
            print("Generating text with Gemini...")
            generated_text = generate_text_with_gemini(mutated_prompt)
            if not generated_text:
                print("Warning: Empty text generated")
                continue
                
            print("Evaluating text with GPT-4...")
            reward = evaluate_text(generated_text)
            scores.append(reward)
            
            if reward > 0.5:
                population.append((mutated_prompt, reward))
            
            if len(population) > 10:
                population = sorted(population, key=lambda x: x[1], reverse=True)[:10]
            
            print(f"Step {step}: Reward={reward:.4f}, Population Size={len(population)}")
            if step % 10 == 0:  # 10ステップごとに現在のベストプロンプトを表示
                best_prompt = max(population, key=lambda x: x[1])[0]
                print(f"Best prompt at step {step}: {best_prompt}")
        
        if scores:
            plt.figure(figsize=(8, 6))
            plt.plot(scores, color='#440154', marker='o', markersize=10, linewidth=2)
            plt.ylim(-0.1, 1.1)  # スコアは0-1の範囲
            plt.xlim(-0.1, len(scores) - 0.9)  # x軸の範囲を調整
            plt.xlabel("ステップ数")
            plt.ylabel("評価スコア")
            plt.title("プロンプトスコアの推移")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # データポイントの値を表示
            for i, score in enumerate(scores):
                plt.annotate(f'{score:.2f}', 
                           (i, score),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center')
            
            output_path = os.path.join(os.getcwd(), 'prompt_optimization_results.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            print("No scores generated to plot")
            
    except Exception as e:
        print(f"Error in optimization process: {str(e)}")
        raise

if __name__ == "__main__":
    train_prompt_optimization()
