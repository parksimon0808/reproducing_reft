import re
import time
from datasets import load_dataset
from math_verify import parse, verify

DATASET_CONFIGS = {
    "gsm8k": ("openai/gsm8k", "main", None),
    "svamp": ("ChilleD/SVAMP", None, None),
    "math": ("qwedsacf/competition_math", None, None),
}

_TEMPLATE_RE = re.compile(r"^\s*<?\s*your answer\s*>?\s*\.?\s*$", re.IGNORECASE)

_UNICODE_TO_LATEX = {
    # Operators and relations
    "\u2212": "-",            # MINUS SIGN
    "\u00d7": "\\times ",     # MULTIPLICATION SIGN
    "\u00b7": "\\cdot ",      # MIDDLE DOT
    "\u2219": "\\cdot ",      # BULLET OPERATOR
    "\u00f7": "\\div ",       # DIVISION SIGN
    "\u00b1": "\\pm ",        # PLUS-MINUS SIGN
    "\u2213": "\\mp ",        # MINUS-OR-PLUS SIGN
    "\u2264": "\\le ",        # LESS-THAN OR EQUAL TO
    "\u2265": "\\ge ",        # GREATER-THAN OR EQUAL TO
    "\u2260": "\\neq ",       # NOT EQUAL TO
    "\u2248": "\\approx ",    # ALMOST EQUAL TO
    "\u2261": "\\equiv ",     # IDENTICAL TO
    "\u226a": "\\ll ",        # MUCH LESS-THAN
    "\u226b": "\\gg ",        # MUCH GREATER-THAN
    "\u221d": "\\propto ",    # PROPORTIONAL TO
    # Set / logic
    "\u222a": "\\cup ",       # UNION
    "\u2229": "\\cap ",       # INTERSECTION
    "\u2208": "\\in ",        # ELEMENT OF
    "\u2209": "\\notin ",     # NOT AN ELEMENT OF
    "\u2282": "\\subset ",    # SUBSET OF
    "\u2286": "\\subseteq ",  # SUBSET OF OR EQUAL TO
    "\u2205": "\\emptyset ",  # EMPTY SET
    "\u2200": "\\forall ",    # FOR ALL
    "\u2203": "\\exists ",    # THERE EXISTS
    "\u00ac": "\\neg ",       # NOT SIGN
    "\u2227": "\\land ",      # LOGICAL AND
    "\u2228": "\\lor ",       # LOGICAL OR
    # Arrows
    "\u2192": "\\to ",        # RIGHTWARDS ARROW
    "\u2190": "\\leftarrow ", # LEFTWARDS ARROW
    "\u21d2": "\\Rightarrow ",# RIGHTWARDS DOUBLE ARROW
    "\u21d4": "\\Leftrightarrow ",  # LEFT RIGHT DOUBLE ARROW
    "\u2194": "\\leftrightarrow ",  # LEFT RIGHT ARROW
    # Special symbols
    "\u221e": "\\infty ",     # INFINITY
    "\u221a": "\\sqrt",       # SQUARE ROOT
    "\u221b": "\\sqrt[3]",    # CUBE ROOT
    "\u03c0": "\\pi ",        # PI
    "\u00b0": "^\\circ",      # DEGREE SIGN
    "\u2220": "\\angle ",     # ANGLE
    "\u27c2": "\\perp ",      # PERPENDICULAR
    "\u2225": "\\parallel ",  # PARALLEL TO
    "\u25b3": "\\triangle ",  # WHITE UP-POINTING TRIANGLE
    "\u2113": "\\ell ",       # SCRIPT SMALL L
    "\u211d": "\\mathbb{R} ", # DOUBLE-STRUCK CAPITAL R
    "\u2124": "\\mathbb{Z} ", # DOUBLE-STRUCK CAPITAL Z
    "\u2115": "\\mathbb{N} ", # DOUBLE-STRUCK CAPITAL N
    "\u2102": "\\mathbb{C} ", # DOUBLE-STRUCK CAPITAL C
    # Greek lowercase
    "\u03b1": "\\alpha ",
    "\u03b2": "\\beta ",
    "\u03b3": "\\gamma ",
    "\u03b4": "\\delta ",
    "\u03b5": "\\epsilon ",
    "\u03b6": "\\zeta ",
    "\u03b7": "\\eta ",
    "\u03b8": "\\theta ",
    "\u03b9": "\\iota ",
    "\u03ba": "\\kappa ",
    "\u03bb": "\\lambda ",
    "\u03bc": "\\mu ",
    "\u03bd": "\\nu ",
    "\u03be": "\\xi ",
    "\u03c1": "\\rho ",
    "\u03c3": "\\sigma ",
    "\u03c4": "\\tau ",
    "\u03c5": "\\upsilon ",
    "\u03c6": "\\phi ",
    "\u03c7": "\\chi ",
    "\u03c8": "\\psi ",
    "\u03c9": "\\omega ",
    # Greek uppercase
    "\u0393": "\\Gamma ",
    "\u0394": "\\Delta ",
    "\u0398": "\\Theta ",
    "\u039b": "\\Lambda ",
    "\u039e": "\\Xi ",
    "\u03a0": "\\Pi ",
    "\u03a3": "\\Sigma ",
    "\u03a6": "\\Phi ",
    "\u03a8": "\\Psi ",
    "\u03a9": "\\Omega ",
    # Superscripts
    "\u2070": "^0",
    "\u00b9": "^1",
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u2074": "^4",
    "\u2075": "^5",
    "\u2076": "^6",
    "\u2077": "^7",
    "\u2078": "^8",
    "\u2079": "^9",
    "\u207a": "^+",
    "\u207b": "^-",
    "\u207f": "^n",
    "\u1d40": "^T",           # MODIFIER LETTER CAPITAL T (transpose)
    # Subscripts
    "\u2080": "_0",
    "\u2081": "_1",
    "\u2082": "_2",
    "\u2083": "_3",
    "\u2084": "_4",
    "\u2085": "_5",
    "\u2086": "_6",
    "\u2087": "_7",
    "\u2088": "_8",
    "\u2089": "_9",
    "\u208a": "_+",
    "\u208b": "_-",
    "\u2099": "_n",
    # Vulgar fractions
    "\u00bd": "\\frac{1}{2}",
    "\u2153": "\\frac{1}{3}",
    "\u2154": "\\frac{2}{3}",
    "\u00bc": "\\frac{1}{4}",
    "\u00be": "\\frac{3}{4}",
    "\u2155": "\\frac{1}{5}",
    "\u2159": "\\frac{1}{6}",
    "\u215b": "\\frac{1}{8}",
}

_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_TO_LATEX))


def _normalize_unicode(text):
    text = _UNICODE_RE.sub(lambda m: _UNICODE_TO_LATEX[m.group()], text)
    # \sqrt(expr) → \sqrt{expr}
    text = re.sub(r"\\sqrt\(([^)]+)\)", r"\\sqrt{\1}", text)
    # \sqrt[n](expr) → \sqrt[n]{expr}
    text = re.sub(r"\\sqrt(\[\d+\])\(([^)]+)\)", r"\\sqrt\1{\2}", text)
    # \sqrt followed by a single non-brace char → \sqrt{x}
    text = re.sub(r"\\sqrt(?!\[)(?!\{)(\S)", r"\\sqrt{\1}", text)
    # \sqrt[n] followed by a single non-brace char → \sqrt[n]{x}
    text = re.sub(r"\\sqrt(\[\d+\])(?!\{)(\S)", r"\\sqrt\1{\2}", text)
    return text


def _extract_boxed(text):
    results = []
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1])
    return results


def load_dataset_split(dataset_name, split="train"):
    path, name, revision = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(path, name=name, split=split, revision=revision)
    questions, golds = [], []
    for row in ds:
        if dataset_name == "gsm8k":
            questions.append(row["question"])
            match = re.search(r"####\s*(.+)", row["answer"])
            golds.append(match.group(1).strip().replace(",", "") if match else row["answer"].strip())
        elif dataset_name == "svamp":
            questions.append(f"{row['Body']} {row['Question']}")
            golds.append(str(row["Answer"]))
        elif dataset_name == "math":
            questions.append(row["problem"])
            matches = _extract_boxed(row["solution"])
            golds.append(matches[-1] if matches else row["solution"].strip())
    return questions, golds


SYSTEM_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then end with \"The answer is <your answer>.\""
)


def format_prompt(question):
    parts = [SYSTEM_INSTRUCTION, ""]
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


def extract_answer(text, dataset_name):
    if not text:
        return None
    # Find ALL "The answer is ..." occurrences; iterate last-to-first so we
    # pick the final (typically cleanest) statement, skipping template echoes.
    matches = list(re.finditer(
        r"[Tt]he answer is\s*[:\s]*(.+?)(?:\.(?:\s|$)|\s*$)",
        text, re.MULTILINE,
    ))
    for m in reversed(matches):
        ans = m.group(1).strip().rstrip(".")
        # Strip paired XML-like tags: <Tag>content</Tag>
        ans = re.sub(r"<([^>]+)>(.*?)</\1>", r"\2", ans).strip()
        # Strip simple angle-bracket wrapper: <answer>
        if ans.startswith("<") and ans.endswith(">") and "<" not in ans[1:-1]:
            ans = ans[1:-1].strip()
        if not ans or _TEMPLATE_RE.match(ans):
            continue
        ans = re.sub(r"(?<=\d),(?=\d)", "", ans)  # strip thousands separators
        ans = _normalize_unicode(ans)
        return ans.replace("$", "").strip()
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    boxed = _extract_boxed(text)
    if boxed:
        return boxed[-1]
    if dataset_name == "mathqa":
        match = re.search(r"\b([a-e])\)?\s*$", text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def verify_answer(prediction, gold, dataset_name):
    if prediction is None:
        return False
    if prediction.strip() == gold.strip():
        return True
    try:
        parsed_gold = parse(f"${gold}$")
        parsed_pred = parse(f"${prediction}$")
        if parsed_gold and parsed_pred:
            if verify(parsed_gold, parsed_pred):
                return True
    except Exception:
        pass
    try:
        return float(prediction.replace(",", "")) == float(gold.replace(",", ""))
    except (ValueError, TypeError):
        pass
    norm_pred = re.sub(r"\s+", "", prediction)
    norm_gold = re.sub(r"\s+", "", gold)
    return norm_pred == norm_gold


def get_pricing(model):
    if model == "gpt-5-mini":
        price_per_1M_input_token, price_per_1M_output_token = 0.25, 2.00
    if model == "gpt-5-nano":
        price_per_1M_input_token, price_per_1M_output_token = 0.05, 0.40

    return price_per_1M_input_token * 1e-6, price_per_1M_output_token * 1e-6


def call_openai(client, model, prompt, temperature, max_tokens, n=1, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                n=n,
            )
            texts = [choice.message.content for choice in response.choices]
            usage = response.usage
            return texts, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            wait = 2 ** attempt
            print(f"API error (attempt {attempt+1}): {e}, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries} retries")
