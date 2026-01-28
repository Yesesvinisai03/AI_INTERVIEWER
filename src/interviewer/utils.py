import json

def safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def last_n_history(history, n=10):
    return history[-n:] if len(history) > n else history
