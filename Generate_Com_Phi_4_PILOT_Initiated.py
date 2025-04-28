import os
import csv
import torch
import random
import gc
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Setup ===
nltk.download("punkt")
torch.cuda.empty_cache()
gc.collect()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/phi-4"

# === Load Model and Tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# === Template Tags (Pilot-side utterances only) ===
pilot_templates = [
    "Requesting descent to <ALT> feet, <FH>.",
    "Inbound to GSO, level <ALT> feet, <FH>.",
    "Requesting vectors for ILS Runway <RWY>, <FH>.",
    "<FH>, 10 miles out, request landing clearance for Runway <RWY>.",
    "Ready for taxi to <DEST>, <FH>.",
    "<FH>, holding short of Runway <RWY>, ready for departure.",
    "<FH>, request frequency change to Ground.",
    "Cleared runway <RWY>, taxiing via <TXWY> to <DEST>."
]

tag_values = {
    "<FH>": ["N466C7", "N123AB", "N345GH", "AAL202", "DAL338", "SWA519"],
    "<TF>": ["118.5", "119.7", "120.1"],
    "<GF>": ["121.9", "122.35", "121.7"],
    "<ALT>": ["8,000", "7,000", "6,500", "10,000"],
    "<NEW_ALT>": ["3,000", "2,500", "4,000", "5,000"],
    "<RWY>": ["23L", "23R", "14", "5"],
    "<HDG>": ["270", "180", "90", "360"],
    "<DIST>": ["8", "10", "5", "3"],
    "<WIND>": ["230 at 10", "250 at 15", "180 at 5"],
    "<TXWY>": ["Bravo", "Charlie", "Delta"],
    "<DEST>": ["Signature", "Atlantic", "Gate 3"],
    "<ROUTE>": ["Bravo, Charlie, hold short Runway 23L", "Alpha, Echo", "Charlie, hold short 23R"]
}

# === Utilities ===
def fill_template(template):
    for tag in tag_values:
        if tag in template:
            template = template.replace(tag, random.choice(tag_values[tag]))
    return template

def compute_bleu(reference, prediction):
    ref_tokens = tokenizer.tokenize(reference)
    pred_tokens = tokenizer.tokenize(prediction)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

rouge = Rouge()

def compute_rouge(reference, prediction):
    try:
        scores = rouge.get_scores(prediction, reference)
        return scores[0]["rouge-l"]["f"]
    except:
        return 0.0

def compute_edit_distance(a, b):
    return nltk.edit_distance(a, b)

def compute_perplexity(text):
    input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

# === Generation + Evaluation ===
print("\n🛠️ Generating Pilot-initiated ATC communications...")
generated_data = []

for _ in range(100):
    pilot_line = fill_template(random.choice(pilot_templates))
    prompt = f"Pilot: {pilot_line}\nATC:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=64,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    atc_response = generated.split("ATC:")[-1].strip() if "ATC:" in generated else generated.strip()

    # Metrics
    bleu = compute_bleu(pilot_line, atc_response)
    rouge_l = compute_rouge(pilot_line, atc_response)
    edit_dist = compute_edit_distance(pilot_line, atc_response)
    perplexity = compute_perplexity(atc_response)

    generated_data.append([
        f"Pilot: {pilot_line}",  # Prompt
        f"ATC: {atc_response}",  # Model Response
        round(bleu, 3),
        round(rouge_l, 3),
        round(edit_dist, 2),
        round(perplexity, 2)
    ])

# === Save Output ===
output_csv = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/phi4_synthetic_conversation_metrics_PilotInitiated.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Pilot Prompt", "Generated ATC Response",
        "BLEU", "ROUGE-L",
        "Edit Distance", "Perplexity"
    ])
    writer.writerows(generated_data)

print(f"\n✅ Saved pilot-initiated dialogue evaluations to: {output_csv}")
