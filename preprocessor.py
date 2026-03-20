import re

ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670]')

ALEF_VARIANTS = re.compile(r'[إأآا]')
TAA_MARBUTA = re.compile(r'ة')
ALEF_MAKSURA = re.compile(r'ى')

def remove_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub('', text)

def normalize_arabic(text: str) -> str:
    text = ALEF_VARIANTS.sub('ا', text)
    text = TAA_MARBUTA.sub('ه', text)
    text = ALEF_MAKSURA.sub('ي', text)
    return text

def clean_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text: str) -> str:
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = clean_whitespace(text)
    return text


if __name__ == "__main__":
    samples = [
        "السَّلَامُ عَلَيْكُمْ",
        "مُحَمَّد",
        "الإسلام والإيمان",
        "أنا أحب القراءة",
        "المملكة العربية السعودية",
        "البرمجة  والتقنية   الحديثة",
    ]

    print("Arabic Text Preprocessor Test")
    print("=" * 50)
    for sample in samples:
        result = preprocess(sample)
        print(f"Input:  {sample}")
        print(f"Output: {result}")
        print("-" * 50)
