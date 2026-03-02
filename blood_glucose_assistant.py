# Required packages and models setup
# conda activate <env-name>
# conda install -c pytorch pytorch torchvision torchaudio cpuonly
# conda install -c conda-forge spacy pytz
# conda install -c conda-forge transformers
# python -m spacy download en_core_web_sm

import re
import sys
from datetime import datetime
import math
import spacy
import pytz
from spacy.matcher import Matcher
from transformers import pipeline
from difflib import SequenceMatcher

class DialogueHistory:
    def __init__(self):
        self.glucose_readings = []  # list of dicts: {time, value, unit, context}
        self.history = []  # all user inputs

    def add_reading(self, value, unit, context):
        now = datetime.now(pytz.timezone('Europe/Stockholm')).isoformat()
        self.glucose_readings.append({
            "time": now,
            "value": value,
            "unit": unit,
            "context": context
        })

    def add_user_input(self, text):
        self.history.append(text)

    def get_recent_readings(self, count=5):
        return self.glucose_readings[-count:]

    def get_full_history(self):
        return self.history

# Load spaCy model and initialize matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Enhanced patterns for context matching
# Fasting patterns
before_meal_pattern = [{"LOWER": "before"}, {"LOWER": {"IN": ["breakfast", "lunch", "dinner", "meal", "eating", "food"]}}]
empty_stomach_pattern = [{"LOWER": "empty"}, {"LOWER": "stomach"}]
fasting_pattern = [{"LOWER": {"IN": ["fasting", "fast"]}}]
pre_meal_pattern = [{"LOWER": {"IN": ["pre", "premeal", "preprandial"]}}]
morning_pattern = [{"LOWER": {"IN": ["morning", "wake", "woke"]}}]
havent_eaten_pattern = [{"LOWER": {"IN": ["haven't", "havent", "not"]}}, {"LOWER": {"IN": ["eaten", "ate", "eat"]}}]

# Postprandial patterns
after_meal_pattern = [{"LOWER": "after"}, {"LOWER": {"IN": ["breakfast", "lunch", "dinner", "meal", "eating", "food"]}}]
post_meal_pattern = [{"LOWER": {"IN": ["post", "postmeal", "postprandial"]}}]
hours_after_pattern = [{"LIKE_NUM": True}, {"LOWER": {"IN": ["hour", "hours", "hr", "hrs"]}}, {"LOWER": "after"}]

# Add all patterns to matcher
matcher.add("FASTING_BEFORE_MEAL", [before_meal_pattern])
matcher.add("FASTING_EMPTY_STOMACH", [empty_stomach_pattern])
matcher.add("FASTING_KEYWORD", [fasting_pattern])
matcher.add("FASTING_PRE_MEAL", [pre_meal_pattern])
matcher.add("FASTING_MORNING", [morning_pattern])
matcher.add("FASTING_HAVENT_EATEN", [havent_eaten_pattern])
matcher.add("POST_AFTER_MEAL", [after_meal_pattern])
matcher.add("POST_KEYWORD", [post_meal_pattern])
matcher.add("POST_HOURS_AFTER", [hours_after_pattern])

# Sentiment classifier for patient emotion
classifier = pipeline("sentiment-analysis")

# Standard unit formats and mappings
STANDARD_UNITS = {
    "mmol/L": ["mmol/l", "mmol", "mm", "mmol/l", "mmol/L", "millimolar"],
    "mg/dL": ["mg/dl", "mgdl", "mg", "mg/dl", "mg/dL", "milligrams", "mg/deciliter"]
}

def standardize_unit(unit):
    """Standardize unit format to consistent representation"""
    if not unit:
        return None
    unit_lower = unit.lower()
    for standard, variants in STANDARD_UNITS.items():
        if any(variant.lower() == unit_lower for variant in variants):
            return standard
    return None

# Enhanced unit keywords list
UNIT_KEYWORDS = ["mmol/l", "mmol", "mg/dl", "mgdl", "mg/deciliter", "millimolar", "mm", "mg"]

# Glucose-related keywords for context-aware extraction
GLUCOSE_KEYWORDS = ["glucose", "blood sugar", "bg", "sugar", "glycemia", "blood glucose", "reading"]

def find_glucose_keywords(doc):
    """Find positions of glucose-related keywords in the document"""
    positions = []
    text_lower = doc.text.lower()
    for keyword in GLUCOSE_KEYWORDS:
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
    return positions

def extract_glucose_value_spacy(text):
    """Extract glucose value with context awareness and validation"""
    doc = nlp(text)
    glucose_positions = find_glucose_keywords(doc)
    
    candidates = []
    
    # Extract all potential numeric values
    # From entities
    for ent in doc.ents:
        if ent.label_ in {"QUANTITY", "CARDINAL"}:
            nums = re.findall(r"\d+(?:\.\d+)?", ent.text)
            for num in nums:
                value = float(num)
                # Calculate distance to nearest glucose keyword
                ent_pos = ent.start_char
                min_distance = float('inf')
                if glucose_positions:
                    min_distance = min(abs(ent_pos - gpos) for gpos in glucose_positions)
                candidates.append((value, min_distance, ent_pos))
    
    # Fallback regex for all numbers
    for match in re.finditer(r"\d+(?:\.\d+)?", text):
        value = float(match.group())
        pos = match.start()
        # Check if this number is already in candidates
        if not any(c[0] == value and abs(c[2] - pos) < 5 for c in candidates):
            min_distance = float('inf')
            if glucose_positions:
                min_distance = min(abs(pos - gpos) for gpos in glucose_positions)
            candidates.append((value, min_distance, pos))
    
    if not candidates:
        return None
    
    # Filter candidates by reasonable glucose range
    valid_candidates = []
    for value, distance, pos in candidates:
        # Check if it's in reasonable range for either unit
        if (1.0 <= value <= 35.0) or (20 <= value <= 600):
            # Check for exclusion patterns (age, time, etc.)
            context_start = max(0, pos - 20)
            context_end = min(len(text), pos + 20)
            context = text[context_start:context_end].lower()
            
            # Exclusion keywords
            exclude_patterns = ["years old", "age", "aged", "am", "pm", "kg", "cm", "pounds", "lbs"]
            if not any(pattern in context for pattern in exclude_patterns):
                valid_candidates.append((value, distance, pos))
    
    if not valid_candidates:
        # If no valid candidates, return the most reasonable from all candidates
        candidates_in_range = [(v, d, p) for v, d, p in candidates if (1.0 <= v <= 35.0) or (20 <= v <= 600)]
        if candidates_in_range:
            return min(candidates_in_range, key=lambda x: x[1])[0]
        return None
    
    # Return the candidate closest to a glucose keyword
    best_candidate = min(valid_candidates, key=lambda x: x[1])
    return best_candidate[0]

def validate_glucose_value(value, unit):
    """Validate if glucose value is in reasonable range for the given unit"""
    if unit == "mmol/L":
        return 1.0 <= value <= 35.0
    elif unit == "mg/dL":
        return 20 <= value <= 600
    return False

def infer_unit_spacy(text):
    """Infer unit from text with standardization"""
    doc = nlp(text)
    
    # Check entity text
    for ent in doc.ents:
        if ent.label_ in {"QUANTITY", "CARDINAL"}:
            txt = ent.text.lower()
            for u in UNIT_KEYWORDS:
                if u in txt:
                    return standardize_unit(u)
    
    # Token-based
    for token in doc:
        tok = token.text.lower()
        if tok in UNIT_KEYWORDS:
            return standardize_unit(tok)
    
    # Full-text check
    txt = text.lower()
    for u in UNIT_KEYWORDS:
        if u in txt:
            return standardize_unit(u)
    
    return None

def validate_unit():
    """Validate unit with standardized format"""
    while True:
        print("Please select the unit:")
        print("<1> mmol/L  <2> mg/dL")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "mmol/L"
        elif choice == "2":
            return "mg/dL"
        else:
            print("Error: Invalid selection. Please enter either 1 or 2.")

# Comprehensive context keywords
FASTING_KEYS = [
    "empty stomach", "fasting", "before meal", "before eating", 
    "haven't eaten", "havent eaten", "not eaten", "pre-meal", 
    "premeal", "preprandial", "ac", "morning", "wake up", 
    "woke up", "before breakfast", "before lunch", "before dinner"
]

POST_KEYS = [
    "after meal", "postprandial", "2 hour", "2 hours", "2-hour",
    "after eating", "post meal", "postmeal", "pc", "pp",
    "after breakfast", "after lunch", "after dinner", "post-meal"
]

def fuzzy_match(text, patterns, threshold=0.7):
    """Fuzzy matching for patterns with spelling tolerance"""
    text_lower = text.lower()
    for pattern in patterns:
        # Direct substring match
        if pattern in text_lower:
            return True
        # Fuzzy match for individual words
        words = pattern.split()
        text_words = text_lower.split()
        for word in words:
            for text_word in text_words:
                similarity = SequenceMatcher(None, word, text_word).ratio()
                if similarity >= threshold:
                    return True
    return False

def infer_context_spacy(text):
    """Infer context with enhanced pattern matching"""
    doc = nlp(text)
    
    # Use matcher for structured patterns
    matches = matcher(doc)
    fasting_matches = ["FASTING_BEFORE_MEAL", "FASTING_EMPTY_STOMACH", 
                       "FASTING_KEYWORD", "FASTING_PRE_MEAL", 
                       "FASTING_MORNING", "FASTING_HAVENT_EATEN"]
    post_matches = ["POST_AFTER_MEAL", "POST_KEYWORD", "POST_HOURS_AFTER"]
    
    for match_id, start, end in matches:
        match_label = nlp.vocab.strings[match_id]
        if match_label in fasting_matches:
            return "fasting"
        elif match_label in post_matches:
            return "postprandial"
    
    # Fallback to keyword matching with fuzzy support
    if fuzzy_match(text, FASTING_KEYS):
        return "fasting"
    if fuzzy_match(text, POST_KEYS):
        return "postprandial"
    
    # Time-based inference
    text_lower = text.lower()
    time_indicators = {
        "fasting": ["6am", "7am", "6:00", "7:00", "early morning"],
        "postprandial": ["1pm", "2pm", "7pm", "8pm", "evening"]
    }
    
    for context, indicators in time_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            return context
    
    return None

def validate_context():
    """Validate context selection"""
    while True:
        print("Please select measurement context:")
        print("[1] Fasting  [2] Postprandial")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "fasting"
        elif choice == "2":
            return "postprandial"
        else:
            print("Error: Invalid selection. Please enter either 1 or 2.")

# Conversion
def convert_to_mmol(value, unit):
    """Convert glucose value to mmol/L"""
    if unit == "mg/dL":
        return round(value / 18.0, 2)
    return value

# Assessment
def assess_insulin_dose(level, context):
    """Assess insulin dose recommendation"""
    if context == "fasting":
        threshold = 7.0
        dose = round(level - threshold, 1)
    else:
        threshold = 11.1
        dose = round(level - threshold, 1)
    if dose >= 1:
        print(f"Consider taking {dose} units of insulin.")
    else:
        print("Glucose slightly elevated; continue monitoring.")

def assess_glucose(level, context, diagnosed, on_insulin):
    """Assess glucose level and provide recommendations"""
    # Low glucose
    if level < 4.0:
        print("Low blood glucose detected (<4.0 mmol/L). Please consume carbohydrates immediately.")
        return
    
    # Normal/high thresholds
    if context == "fasting":
        normal_low, normal_high = 4.0, 7.0
    else:
        normal_low, normal_high = 4.0, 11.1
    
    if normal_low <= level <= normal_high:
        if diagnosed:
            print("Blood glucose well controlled.")
        else:
            print("Blood glucose within normal range.")
    else:
        if not diagnosed:
            print("Blood glucose abnormal; please consult a healthcare professional.")
        else:
            if on_insulin:
                assess_insulin_dose(level, context)
            else:
                print("Blood glucose elevated; consider lifestyle and diet adjustments.")

# Trend analysis
def analyze_trends(history, current_val, context):
    """Analyze glucose trends from historical data"""
    relevant = [r["value"] for r in history if r["context"] == context]
    if len(relevant) >= 3:
        avg = sum(relevant[-3:]) / 3
        diff = round(current_val - avg, 1)
        if diff > 1.0:
            print(f"Current value is {diff} mmol/L above the average of last 3 readings.")
        elif diff < -1.0:
            print(f"Current value is {abs(diff)} mmol/L below the average of last 3 readings.")
        else:
            print(f"Current value is close to the average ({round(avg,1)} mmol/L).")

# Main chat
def start_chat():
    """Main chat interface"""
    history = DialogueHistory()
    print("Welcome to the Blood Glucose Assistant. Type 'exit' to quit.")
    
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        
        history.add_user_input(text)
        low = text.lower()
        
        if low in {"exit", "quit"}:
            print("Goodbye!")
            sys.exit()
        
        if "show full history" in low:
            for idx, entry in enumerate(history.get_full_history(), 1):
                print(f"{idx}: {entry}")
            continue
        
        # Emotion analysis
        emo = classifier(text)[0]
        print(f"Patient emotion: {emo['label']} (score {emo['score']:.2f})")
        
        # Extraction with improved context awareness
        value = extract_glucose_value_spacy(text)
        if value is None:
            print("Could not detect glucose value. Please re-enter.")
            continue
        
        # Unit inference with standardization
        unit = infer_unit_spacy(text)
        if unit is None:
            unit = validate_unit()
        
        # Validate the extracted value with the unit
        if not validate_glucose_value(value, unit):
            print(f"The value {value} {unit} seems unusual for blood glucose.")
            confirm = input("Is this correct? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Please re-enter your glucose reading.")
                continue
        
        # Context inference with enhanced patterns
        context = infer_context_spacy(text)
        if context is None:
            context = validate_context()
        
        # Conversion
        value_mmol = convert_to_mmol(value, unit)
        
        # Ask about diagnosis and insulin usage
        while True:
            diag_input = input("Has the patient been diagnosed with diabetes? (y/n): ").strip().lower()
            if diag_input in ['y', 'n']:
                diag = diag_input == 'y'
                break
            else:
                print("Error: Invalid input. Please enter 'y' or 'n'.")
        
        insulin = False
        if diag:
            while True:
                insulin_input = input("Is the patient on insulin? (y/n): ").strip().lower()
                if insulin_input in ['y', 'n']:
                    insulin = insulin_input == 'y'
                    break
                else:
                    print("Error: Invalid input. Please enter 'y' or 'n'.")
        
        # Assessment
        assess_glucose(value_mmol, context, diag, insulin)
        
        # Record with standardized unit
        history.add_reading(value_mmol, "mmol/L", context)
        
        # Display recent
        recent = history.get_recent_readings(5)
        print("Recent readings:")
        for r in recent:
            formatted_time = r["time"].replace("T", " ").split(".")[0]
            print(f"{formatted_time}: {r['value']} mmol/L ({r['context']})")
        
        # Trend analysis
        analyze_trends(history.glucose_readings, value_mmol, context)

if __name__ == "__main__":
    start_chat()