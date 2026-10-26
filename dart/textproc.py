"""Post-processing of generated AI-side texts, identical to the processing of the CATS AI texts.

fmt / clean_ai / cut: format unification, AI-side delete-only cleanup of delivery phrases and
disclaimers, and the word cut that gives both texts of a pair the same length (shortcut_trim_v1).
normalize: label-independent typography normalisation (typo_fix_v1, the D00 path).
"""
import re

WORD = re.compile(r"\S+")


def fmt(t):
    t = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$", " ", t)
    t = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", t)
    t = re.sub(r"(?m)^[ \t]*[•▪●*-][ \t]+(?=\S)", "", t)
    t = t.replace("**", "")
    t = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])|(?:(?<=\s)|^)\*(?=\s|$)", "", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()
LEAD = re.compile(r"(?i)^\s*(?:(?:okay|ok|sure|certainly|absolutely|of course)[!.,]+\s+)?(?:(?:here(?:'s| is| are)|below is)\b[^.:!?]{0,150}[:.!]\s+)?")
INTJ = re.compile(r"(?i)^\s*(?:okay|ok|sure|certainly|absolutely)[!.,]+\s+(?:so[,]?\s+)?")
TAILS = re.compile(r"(?i)(not (?:constitute )?(?:professional )?(?:financial|investment|legal|tax|medical)(?: or (?:legal|tax|financial))? advice|consult (?:a|with a|your)\s+(?:qualified|financial|licensed|professional|tax|medical)|for (?:general )?(?:informational|educational) purposes|disclaimer|let me know|hope this helps|feel free to|i hope this)")
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
def clean_ai(t):
    t2 = LEAD.sub("", t, count=1); t2 = INTJ.sub("", t2, count=1)
    if t2 and t2[0].islower(): t2 = t2[0].upper() + t2[1:]
    s = SPLIT.split(t2)
    while len(s) > 1 and TAILS.search(s[-1]): s.pop()
    return " ".join(s).strip()
def cut(t, W):
    w = list(WORD.finditer(t)); return t if len(w) <= W else t[:w[W - 1].end()]


KEEP=re.compile(r'(?P<fence>^[ \t]*(?:`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$))'
                r'|(?P<inline>(?P<ticks>`+)(?!`)[^\n`]{1,80}?(?<!`)(?P=ticks)(?!`))'
                r'|(?P<math>\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))'
                r'|(?P<code_line>^[^\n]*(?:printf|print\(|echo |%d|%s|std::|console\.|System\.out|\bchar\b|\\t)[^\n]*$)',re.M)
def _ws(t):
    t=re.sub(r'[ \t]{2,}',' ',t)
    t=re.sub(r'(?m)^[ \t]+|[ \t]+$','',t)
    return t.strip()
def literal_newlines(t,c):
    keep=[(m.start(),m.end()) for m in KEEP.finditer(t)]
    out=[];last=0
    for m in re.finditer(r'(?<!\\)(?:\\n)+',t):
        s,e=m.span()
        if any(a<=s<b for a,b in keep):continue
        tok_start=max(t.rfind(' ',0,s),t.rfind('\n',0,s))+1
        tok=t[tok_start:s]
        nxt=t[e:e+1]
        if re.match(r'[A-Za-z]:\\|\\\\',tok) and nxt.islower():
            c['literal_newline_kept_path']+=1;continue
        out.append(t[last:s]);out.append(' ');last=e;c['literal_newline']+=1
    out.append(t[last:])
    return ''.join(out)
def typography(t,c):
    def sub(p,r,t,k):
        t,n=re.subn(p,r,t)
        if n:c[k]+=n
        return t
    t=sub('[‘’‚‛′‹›]',"'",t,'single_quote')
    t=sub('[“”„‟″«»]','"',t,'double_quote')
    t=sub('[‐‑−]','-',t,'hyphen_minus_variant')
    t=sub(r'(?<=\d)[ \t]*[‒–—―]+[ \t]*(?=\d)','-',t,'numeric_range_dash')
    t=sub(r'(?<=\w)[‒–](?=\w)','-',t,'compound_en_dash')
    t=sub(r'[ \t]*[‒–—―⸺⸻]+[ \t]*',' - ',t,'spaced_dash')
    t=sub('…','...',t,'ellipsis')
    return t


def normalize(text: str) -> str:
    """Literal newlines, typography and whitespace; returns "" when nothing is left."""
    from collections import Counter
    c = Counter()
    t = literal_newlines(text, c)
    t = typography(t, c)
    return _ws(t)
