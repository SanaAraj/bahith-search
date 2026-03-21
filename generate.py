import re
from typing import List, Dict, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

_client = None

SYSTEM_PROMPT = """أنت مساعد بحث ذكي. أجب على السؤال باللغة العربية فقط بناءً على السياق المقدم.

قواعد:
- استخدم المعلومات من السياق فقط
- كن موجزاً (2-4 جمل)
- إذا لم يحتوي السياق على معلومات كافية، قل ذلك
- لا تختلق معلومات

بعد الإجابة، أضف:
CONFIDENCE: [رقم من 1-10 يعبر عن مدى ثقتك في الإجابة]
RELATED: [3 أسئلة مرتبطة قد يريد المستخدم طرحها، مفصولة بـ |]"""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def generate_answer(query: str, results: List[Dict]) -> Dict:
    default = {"answer": None, "confidence": 0, "related": []}

    if not results:
        return default

    if not OPENAI_API_KEY:
        return default

    context_parts = []
    for i, r in enumerate(results[:3], 1):
        context_parts.append(f"[{i}] {r['title']}:\n{r['content'][:500]}")

    context = "\n\n".join(context_parts)

    user_message = f"""السياق:
{context}

السؤال: {query}

الإجابة:"""

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=400
        )
        text = response.choices[0].message.content.strip()

        answer = text
        confidence = 7
        related = []

        if "CONFIDENCE:" in text:
            parts = text.split("CONFIDENCE:")
            answer = parts[0].strip()
            rest = parts[1] if len(parts) > 1 else ""

            conf_match = re.search(r'(\d+)', rest)
            if conf_match:
                confidence = min(10, max(1, int(conf_match.group(1))))

            if "RELATED:" in rest:
                related_part = rest.split("RELATED:")[1].strip()
                related = [r.strip() for r in related_part.split("|") if r.strip()][:3]

        return {"answer": answer, "confidence": confidence, "related": related}

    except Exception as e:
        print(f"Generation error: {e}")
        return default


def get_source_citations(results: List[Dict], count: int = 3) -> str:
    sources = [r['title'] for r in results[:count]]
    return "، ".join(sources)


if __name__ == "__main__":
    test_results = [
        {
            'title': 'الذكاء الاصطناعي',
            'content': 'الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم بإنشاء أنظمة قادرة على أداء مهام تتطلب ذكاء بشرياً. يشمل التعلم الآلي ومعالجة اللغات الطبيعية والرؤية الحاسوبية.',
            'score': 0.95
        },
        {
            'title': 'التعلم الآلي',
            'content': 'التعلم الآلي هو فرع من الذكاء الاصطناعي يركز على بناء أنظمة تتعلم من البيانات وتحسن أداءها بمرور الوقت دون برمجة صريحة.',
            'score': 0.88
        }
    ]

    query = "ما هو الذكاء الاصطناعي؟"
    print(f"Query: {query}")
    print("\nGenerating answer...")

    answer = generate_answer(query, test_results)
    if answer:
        print(f"\nAnswer: {answer}")
        print(f"Sources: {get_source_citations(test_results)}")
    else:
        print("Failed to generate answer (API key may be missing)")
