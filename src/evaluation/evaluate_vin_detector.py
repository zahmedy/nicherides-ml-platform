import re

VIN_REGEX = re.compile(r"[A-HJ-NPR-Z0-9]{17}")

def clean_vin(text: str):
    text = text.upper()
    text = text.replace(" ", "").replace("-", "")
    #text = text.replace("O", "0")  # optional, be careful
    #text = text.replace("I", "1")  # optional, be careful

    matches = VIN_REGEX.findall(text)
    return matches[0] if matches else None
