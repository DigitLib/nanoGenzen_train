"""
nanogentzen/parser.py
Compiles symbolic expressions and natural language deduction prompts
into formal Gentzen Sequents.
"""

import re
from typing import List, Optional, Tuple
from nanogentzen.kernel import And, Formula, Imp, Not, Or, Sequent, Var


class FormulaParser:
    """Recursive descent parser for propositional formulas."""

    def __init__(self, text: str):
        # Normalize operator symbols
        text = text.replace("⟶", "|-").replace("->", "=>").replace("∧", "&").replace("∨", "|").replace("¬", "~")
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> List[str]:
        # NOTE: TURNSTILE must come before OR so '|-' is not parsed as '|' and '-'
        token_spec = [
            ("TURNSTILE", r"\|-"),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("IMP", r"=>"),
            ("AND", r"&|\band\b"),
            ("OR", r"\||\bor\b"),
            ("NOT", r"~|\bnot\b"),
            ("COMMA", r","),
            ("ZERO", r"\b0\b|\bfalse\b"),
            ("VAR", r"[A-Za-z_][A-Za-z0-9_]*"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, text, re.IGNORECASE):
            kind = mo.lastgroup
            val = mo.group()
            if kind == "SKIP":
                continue
            if kind == "AND":
                tokens.append("&")
            elif kind == "OR":
                tokens.append("|")
            elif kind == "NOT":
                tokens.append("~")
            elif kind == "IMP":
                tokens.append("=>")
            elif kind == "TURNSTILE":
                tokens.append("|-")
            else:
                tokens.append(val)
        return tokens

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError(f"Unexpected end of input, expected {expected}")
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}' at token {self.pos}")
        self.pos += 1
        return tok

    def parse_sequent(self) -> Sequent:
        """Parses Gamma |- Delta"""
        gamma: List[Formula] = []
        delta: List[Formula] = []

        # Parse antecedents (Gamma)
        if self._peek() and self._peek() != "|-":
            while True:
                if self._peek() in ("0", "false"):
                    self._consume()
                else:
                    gamma.append(self.parse_formula())
                if self._peek() == ",":
                    self._consume(",")
                else:
                    break

        if self._peek() == "|-":
            self._consume("|-")
        elif not gamma and not self._peek():
            raise ValueError("Empty sequent")

        # Parse succedents (Delta)
        if self._peek():
            if self._peek() in ("0", "false"):
                self._consume()
            else:
                while self._peek():
                    delta.append(self.parse_formula())
                    if self._peek() == ",":
                        self._consume(",")
                    else:
                        break

        return Sequent(tuple(gamma), tuple(delta))

    def parse_formula(self) -> Formula:
        return self._parse_imp()

    def _parse_imp(self) -> Formula:
        left = self._parse_or()
        if self._peek() == "=>":
            self._consume("=>")
            right = self._parse_imp()  # Right-associative
            return Imp(left, right)
        return left

    def _parse_or(self) -> Formula:
        node = self._parse_and()
        while self._peek() == "|":
            self._consume("|")
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> Formula:
        node = self._parse_not()
        while self._peek() == "&":
            self._consume("&")
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> Formula:
        if self._peek() == "~":
            self._consume("~")
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Formula:
        tok = self._peek()
        if tok == "(":
            self._consume("(")
            expr = self.parse_formula()
            self._consume(")")
            return expr
        elif tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
            self._consume()
            return Var(tok)
        raise ValueError(f"Unexpected token in formula: {tok}")


def parse_symbolic_sequent(text: str) -> Optional[Sequent]:
    """Tries to parse a symbolic sequent like '(P => Q), P |- Q'."""
    # A symbolic sequent or formula must contain formal operators or turnstile
    if not any(sym in text for sym in ["|-", "⟶", "=>", "->", "&", "|", "~"]):
        return None
    try:
        parser = FormulaParser(text)
        seq = parser.parse_sequent()
        if parser.pos < len(parser.tokens):
            return None
        return seq
    except Exception:
        return None


def normalize_word(w: str) -> str:
    w = w.lower()
    if w in {"is", "are", "in", "the", "a", "an", "it", "did", "do", "does", "were", "was", "then", "of", "to", "there"}:
        return ""
    if w.endswith("ing") and len(w) > 4:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 3:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 3:
        w = w[:-2]
    elif w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
        w = w[:-1]
    return w.capitalize()


def clean_term(phrase: str) -> Var:
    """Cleans and normalizes a natural language phrase into a consistent PascalCase propositional variable."""
    words = [normalize_word(w) for w in re.findall(r"[A-Za-z0-9]+", phrase)]
    words = [w for w in words if w]
    name = "".join(words)
    return Var(name if name else "X")


def parse_natural_language(text: str) -> Optional[Tuple[Sequent, str]]:
    """
    Translates natural language syllogisms, implications, and queries
    into intuitionistic sequents with an explanation.
    """
    text_clean = text.strip()
    
    # Extract questions (ending with ?) or last sentence
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text_clean) if s.strip()]
    if not sentences:
        return None

    # Identify conclusion / goal
    has_q = "?" in text_clean
    target_q = sentences[-1]
    premise_sentences = sentences[:-1] if (has_q or len(sentences) > 1) else sentences

    premises: List[Formula] = []

    for s in premise_sentences:
        lower_s = s.lower().strip()
        
        # 1. Implication with 'if' ... 'then' or 'if' ... ','
        if lower_s.startswith("if "):
            content = s[3:].strip()
            parts = re.split(r"\bthen\b|,", content, maxsplit=1)
            if len(parts) == 2:
                premises.append(Imp(clean_term(parts[0]), clean_term(parts[1])))
                continue
            # "If A is in B"
            parts_in = re.split(r"\bis\s+in\b|\bis\b", content, maxsplit=1, flags=re.IGNORECASE)
            if len(parts_in) == 2:
                premises.append(Imp(clean_term(parts_in[0]), clean_term(parts_in[1])))
                continue

        # 2. "In X, Y" (e.g. "In Spring, flowers blooming" or "In Spring flowers blooming")
        if lower_s.startswith("in "):
            content = s[3:].strip()
            parts = re.split(r",|\bare\b|\bis\b", content, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                premises.append(Imp(clean_term(parts[0]), clean_term(parts[1])))
                continue
            words = content.split(maxsplit=1)
            if len(words) == 2:
                premises.append(Imp(clean_term(words[0]), clean_term(words[1])))
                continue

        # 3. "X implies Y" / "X leads to Y" / "X means Y"
        match_imp = re.split(r"\bimplies\b|\bleads to\b|\bmeans\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(match_imp) == 2:
            premises.append(Imp(clean_term(match_imp[0]), clean_term(match_imp[1])))
            continue

        # 4. Negation: "Not X" / "It is not wet" / "~Wet"
        if lower_s.startswith("not ") or " not " in lower_s or "didn't" in lower_s or "no " in lower_s:
            # Extract negated subject
            core = re.sub(r"\bnot\b|\bdid not\b|\bdidn't\b|\bis not\b|\bno\b", "", s, flags=re.IGNORECASE)
            premises.append(Not(clean_term(core)))
            continue

        # 5. Simple Assertion: "X is in Y" -> X => Y
        parts_is = re.split(r"\bis\s+in\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts_is) == 2:
            premises.append(Imp(clean_term(parts_is[0]), clean_term(parts_is[1])))
            continue

        # Default atomic variable
        premises.append(clean_term(s))

    # Parse Goal from target_q
    q_lower = target_q.lower()
    
    # Syllogism query: "Are flowers blooming in April?" -> April => FlowersBlooming
    match_q_in = re.search(r"\b(?:are|is|do|did)\s+(.+?)\s+in\s+([a-zA-Z0-9]+)", target_q, re.IGNORECASE)
    if match_q_in:
        property_term = clean_term(match_q_in.group(1))
        subject_term = clean_term(match_q_in.group(2))
        goal = Imp(subject_term, property_term)
    elif "not " in q_lower or "didn't" in q_lower:
        core = re.sub(r"\b(?:are|is|did|do|was|were)\s+|\bnot\b|\?", "", target_q, flags=re.IGNORECASE)
        goal = Not(clean_term(core))
    else:
        goal = clean_term(target_q)

    seq = Sequent(tuple(premises), (goal,))
    desc = f"Extracted {len(premises)} premise(s) ⟶ Goal: {goal.to_str()}"
    return seq, desc
