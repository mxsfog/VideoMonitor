"""Таксономия признаков риска для учебного pipeline.

Словари связывают текстовые и визуальные признаки с подклассами REST-контракта.
Возрастные рамки используются как инженерная ориентация, а не как юридическое
заключение.
"""

from __future__ import annotations

import re
from typing import Any

OFFICIAL_AGE_MARKS = ("0+", "6+", "12+", "16+", "18+")

AGE_RATING_RULES: dict[str, dict[str, Any]] = {
    "0+": {
        "official": True,
        "legal_basis": "436-FZ article 7",
        "summary": (
            "Content for children under 6; may contain only episodic "
            "non-naturalistic non-sexual violence when the genre/plot "
            "condemns violence or shows compassion to the victim."
        ),
        "allowed_criteria": [
            "no harmful information",
            "episodic non-naturalistic non-sexual violence with condemnation or compassion",
        ],
    },
    "3+": {
        "official": False,
        "alias_for": "0+",
        "legal_basis": "internal compatibility alias, not a 436-FZ age mark",
        "summary": "Internal preschool profile; legally treated as 0+.",
        "allowed_criteria": [
            "same constraints as 0+",
        ],
    },
    "6+": {
        "official": True,
        "legal_basis": "436-FZ article 8",
        "summary": "0+ plus limited non-naturalistic illness, accident, disaster, death, crime.",
        "allowed_criteria": [
            "short non-naturalistic illness, except severe illness, without humiliation",
            "non-naturalistic accident, disaster or non-violent death without shown consequences",
            "episodic anti-social or criminal actions if not justified and condemned",
        ],
    },
    "12+": {
        "official": True,
        "legal_basis": "436-FZ article 9",
        "summary": (
            "6+ plus episodic non-naturalistic violence, limited mentions of "
            "substances/gambling and non-erotic heterosexual relations."
        ),
        "allowed_criteria": [
            "episodic non-naturalistic cruelty or violence without shown killing or mutilation",
            "episodic mention, without demonstration, of alcohol, tobacco, drugs, gambling",
            "non-erotic, non-offensive, non-naturalistic sexual relations between man and woman",
        ],
    },
    "16+": {
        "official": True,
        "legal_basis": "436-FZ article 10",
        "summary": (
            "12+ plus non-naturalistic death/disaster, non-naturalistic violence, "
            "drug information without demonstration and separate non-obscene swear words."
        ),
        "allowed_criteria": [
            "death, disaster, accident or illness without naturalistic consequences",
            "cruelty or violence without naturalistic killing or mutilation and with condemnation",
            "drug information without demonstration and with negative attitude",
            "separate swear words that are not obscene",
            "non-offensive sexual relations between man and woman except sexual acts",
        ],
    },
    "18+": {
        "official": True,
        "legal_basis": "436-FZ article 5 part 2 and article 12",
        "summary": "Information prohibited for children.",
        "prohibited_criteria": [
            "inducement to actions threatening life or health, including self-harm or suicide",
            "content causing desire to use drugs, tobacco, nicotine products, intoxicants, alcohol",
            "content causing desire to gamble, engage in prostitution, vagrancy or begging",
            "justification of violence or cruelty, or inducement to violence",
            "sexual violence",
            "humiliation of dignity and public morality with unlawful or violent actions",
            "denial of family values or disrespect for parents/family",
            "propaganda or demonstration of nontraditional sexual relations or preferences",
            "pedophilia propaganda",
            "content capable of causing desire to change sex",
            "propaganda of refusal to have children",
            "justification of unlawful behavior",
            "obscene language",
            "pornography",
            "identifying information about a minor victim of unlawful actions",
            "information product made by a foreign agent",
        ],
    },
}


RISK_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "life_health_self_harm_suicide",
        "age_mark": "18+",
        "contract_class": "DEVIANT",
        "subclass": "SUICIDE",
        "legal_basis": "436-FZ article 5 part 2 item 1",
        "description": "Inducement to self-harm or suicide.",
        "terms": ["самоубийство", "суицид", "смертельный челлендж", "self harm", "suicide"],
        "patterns": [r"(?<!\w)(убей\s+себя|покончи\s+с\s+собой)(?!\w)"],
    },
    {
        "id": "threat_to_life_or_health",
        "age_mark": "18+",
        "contract_class": "DEVIANT",
        "subclass": "VIOLENCE",
        "legal_basis": "436-FZ article 5 part 2 items 1 and 3",
        "description": "Threats, weapons, inducement to violence, justification of violence.",
        "terms": [
            "оружие",
            "огнестрельное оружие",
            "холодное оружие",
            "пистолет",
            "револьвер",
            "винтовка",
            "нож",
            "кровь",
            "насилие",
            "жестокость",
            "убить",
            "убью",
            "убей",
            "убивать",
            "стрелять",
            "ствол",
            "weapon",
            "gun",
            "handgun",
            "pistol",
            "revolver",
            "firearm",
            "rifle",
            "knife",
            "blood",
            "violence",
            "kill",
            "shoot",
        ],
        "patterns": [
            r"(?<!\w)(пистолет\w*|револьвер\w*|винтовк\w*|оружи\w*|стрел\w*|кров\w*|насили\w*|жесток\w*)(?!\w)",
            r"\b(handgun|pistol|revolver|firearm|gun|rifle|weapon|knife|blood|violence)\b",
        ],
    },
    {
        "id": "drugs_and_intoxicants",
        "age_mark": "18+",
        "contract_class": "DRUGS",
        "subclass": "DRUGS",
        "legal_basis": "436-FZ article 5 part 2 item 2",
        "description": "Drugs, psychotropic and other intoxicating substances.",
        "terms": [
            "наркотик",
            "наркотики",
            "психотроп",
            "одурманивающее вещество",
            "drugs",
            "drug",
        ],
        "patterns": [
            r"(?<!\w)(наркотик\w*|психотроп\w*|одурман\w*)(?!\w)",
            r"\b(drugs?)\b",
        ],
    },
    {
        "id": "tobacco_nicotine_products",
        "age_mark": "18+",
        "contract_class": "DRUGS",
        "subclass": "SMOKING",
        "legal_basis": "436-FZ article 5 part 2 item 2",
        "description": "Tobacco, nicotine products and smoking inducement.",
        "terms": [
            "табак",
            "сигарета",
            "вейп",
            "никотин",
            "tobacco",
            "cigarette",
            "vape",
            "nicotine",
            "smoking",
        ],
        "patterns": [
            r"(?<!\w)(табак\w*|сигарет\w*|вейп\w*|никотин\w*|курени\w*)(?!\w)",
            r"\b(tobacco|cigarettes?|vape|nicotine|smoking)\b",
        ],
    },
    {
        "id": "alcohol_products",
        "age_mark": "18+",
        "contract_class": "DRUGS",
        "subclass": "ALCOHOL",
        "legal_basis": "436-FZ article 5 part 2 item 2",
        "description": "Alcohol product inducement.",
        "terms": [
            "алкоголь",
            "пиво",
            "водка",
            "alcohol",
            "beer",
            "vodka",
        ],
        "patterns": [
            r"(?<!\w)(алкогол\w*|пив\w*|водк\w*)(?!\w)",
            r"\b(alcohol|beer|vodka)\b",
        ],
    },
    {
        "id": "gambling_prostitution_vagrancy_begging",
        "age_mark": "18+",
        "contract_class": "LUDOMANIA",
        "subclass": "LUDOMANIA",
        "legal_basis": "436-FZ article 5 part 2 item 2",
        "description": "Gambling and related anti-social inducement.",
        "terms": ["казино", "азартная игра", "ставки", "букмекер", "проституция", "бродяжничество", "попрошайничество", "casino", "gambling", "betting"],
        "patterns": [r"(?<!\w)(казино|азартн\w+|ставк\w*|букмекер\w*|проституц\w*|бродяж\w*|попрошайн\w*)(?!\w)", r"\b(casino|gambling|betting)\b"],
    },
    {
        "id": "sexual_violence_pornography",
        "age_mark": "18+",
        "contract_class": "SEX",
        "subclass": "SEX",
        "legal_basis": "436-FZ article 5 part 2 items 3.1 and 7",
        "description": "Sexual violence and pornography.",
        "terms": ["сексуальное насилие", "порнография", "порно", "sexual violence", "pornography", "porn"],
        "patterns": [r"(?<!\w)(сексуальн\w+\s+насили\w*|порнограф\w*|порно)(?!\w)", r"\b(sexual\s+violence|pornography|porn)\b"],
    },
    {
        "id": "obscene_or_abusive_language",
        "age_mark": "18+",
        "contract_class": "DEVIANT",
        "subclass": "OBSCENE_LANGUAGE",
        "legal_basis": "436-FZ article 5 part 2 item 6 and part 3 item 4",
        "description": "Obscene language and abusive non-obscene expressions.",
        "terms": ["нецензурная брань", "брань", "мат", "оскорбление", "ублюдок", "ублюдки", "obscene language", "swear word"],
        "patterns": [r"(?<!\w)(бран\w*|мат|оскорблен\w*|ублюд\w*)(?!\w)", r"\b(obscene|swear\s+word|abusive\s+language)\b"],
    },
    {
        "id": "family_values_nontraditional_and_sex_change",
        "age_mark": "18+",
        "contract_class": "ANTITRADITIONAL",
        "subclass": "LGBT",
        "legal_basis": "436-FZ article 5 part 2 items 4, 4.1 and 4.3",
        "description": "Restricted nontraditional relations/preferences and sex-change propaganda categories.",
        "terms": ["нетрадиционные отношения", "нетрадиционные сексуальные отношения", "смена пола"],
        "patterns": [
            r"(?<!\w)(нетрадиционн\w+\s+(?:сексуальн\w+\s+)?отношен\w*|смен\w+\s+пола)(?!\w)"
        ],
    },
    {
        "id": "childfree_propaganda",
        "age_mark": "18+",
        "contract_class": "ANTITRADITIONAL",
        "subclass": "CHILDFREE",
        "legal_basis": "436-FZ article 5 part 2 item 4.4",
        "description": "Propaganda of refusal to have children.",
        "terms": ["чайлдфри", "отказ от деторождения", "childfree"],
        "patterns": [
            r"(?<!\w)(чайлдфри|отказ\w*\s+от\s+деторожд\w*)(?!\w)",
            r"\bchildfree\b",
        ],
    },
    {
        "id": "pedophilia",
        "age_mark": "18+",
        "contract_class": "SEX",
        "subclass": "KIDSPORN",
        "legal_basis": "436-FZ article 5 part 2 item 4.2",
        "description": "Pedophilia propaganda.",
        "terms": ["педофилия", "pedophilia"],
        "patterns": [r"(?<!\w)(педофил\w*)(?!\w)", r"\b(pedophilia|pedophile)\b"],
    },
    {
        "id": "unlawful_behavior_and_minor_victim_data",
        "age_mark": "18+",
        "contract_class": "DEVIANT",
        "subclass": "VANDALISM",
        "legal_basis": "436-FZ article 5 part 2 items 3.2, 5 and 8",
        "description": "Unlawful behavior justification, violent unlawful acts, identifying minor-victim data.",
        "terms": ["противоправное поведение", "хулиганство", "вандализм", "несовершеннолетний пострадавший", "vandalism"],
        "patterns": [r"(?<!\w)(противоправн\w+|хулиган\w*|вандал\w*|несовершеннолетн\w+\s+пострадавш\w*)(?!\w)", r"\b(vandalism|unlawful\s+behavior)\b"],
    },
    {
        "id": "terrorism_content",
        "age_mark": "18+",
        "contract_class": "TERRORISM",
        "subclass": "TERROR",
        "legal_basis": "436-FZ article 5 part 2 items 1, 3 and 5",
        "description": "Terrorism-related content or inducement.",
        "terms": [
            "террор",
            "терроризм",
            "terror",
            "terrorism",
        ],
        "patterns": [
            r"(?<!\w)террор\w*(?!\w)",
            r"\bterroris[mt]\b",
        ],
    },
    {
        "id": "extremism_nazi_symbols",
        "age_mark": "18+",
        "contract_class": "TERRORISM",
        "subclass": "EXTREMISM",
        "legal_basis": "436-FZ article 5; 114-FZ article 1; КоАП РФ article 20.3 context",
        "description": "Extremism and nazi/fascist attributes or symbols.",
        "terms": [
            "экстремизм",
            "свастика",
            "нацистская символика",
            "нацистская форма",
            "нацист",
            "гитлер",
            "фашизм",
            "форма сс",
            "символика сс",
            "der stürmer",
            "der sturmer",
            "extremism",
            "swastika",
            "nazi",
            "nazi symbol",
            "nazi uniform",
            "ss uniform",
            "hitler",
            "fascism",
        ],
        "patterns": [
            r"der\s+st[uü]rmer",
            r"(?<!\w)(экстреми\w*|свастик\w*|наци\w*|гитлер\w*|фаши\w*|хайль|символик\w+\s+сс|форм\w+\s+сс)(?!\w)",
            r"\b(extremis[mt]|swastika|nazi|hitler|fascis[mt]|ss\s+uniform)\b",
        ],
    },
    {
        "id": "foreign_agent_product",
        "age_mark": "18+",
        "contract_class": "ANTIPATRIOTIC",
        "subclass": "INOAGENTCONTENT",
        "legal_basis": "436-FZ article 5 part 2 item 9",
        "description": "Information product made by a foreign agent.",
        "terms": ["иностранный агент", "иноагент", "foreign agent"],
        "patterns": [r"(?<!\w)(иностранн\w+\s+агент\w*|иноагент\w*)(?!\w)", r"\bforeign\s+agent\b"],
    },
)


SUBCLASS_TO_CLASS: dict[str, str] = {
    str(rule["subclass"]): str(rule["contract_class"])
    for rule in RISK_RULES
}


_RISK_PATTERNS: tuple[tuple[re.Pattern[str], dict[str, Any]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), rule)
    for rule in RISK_RULES
    for pattern in rule["patterns"]
)

_TERM_LOOKUP: dict[str, dict[str, Any]] = {
    str(term).lower(): rule
    for rule in RISK_RULES
    for term in rule["terms"]
}


def normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def iter_risk_matches(text: object) -> list[dict[str, Any]]:
    """Найти уникальные риск-признаки в тексте или названии объекта."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    exact = _TERM_LOOKUP.get(normalized)
    if exact:
        key = (normalized, str(exact["subclass"]))
        seen.add(key)
        matches.append({"match": normalized, **exact})

    for pattern, rule in _RISK_PATTERNS:
        for found in pattern.finditer(normalized):
            token = found.group(0)
            key = (token, str(rule["subclass"]))
            if key in seen:
                continue
            seen.add(key)
            matches.append({"match": token, **rule})

    return matches


def keyword_matches(text: object) -> list[tuple[str, str]]:
    """Совместимый формат для старых частей pipeline."""
    return [
        (str(match["match"]), str(match["subclass"]))
        for match in iter_risk_matches(text)
    ]


def label_to_subclass(label: object) -> str | None:
    """Сопоставить текстовую метку с подклассом контракта."""
    normalized = normalize_text(label)
    if not normalized:
        return None
    if normalized.upper() in SUBCLASS_TO_CLASS:
        return normalized.upper()
    matches = iter_risk_matches(normalized)
    if not matches:
        return None
    return str(matches[0]["subclass"])
