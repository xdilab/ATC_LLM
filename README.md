# **ATC_LLM**

# **Gen-Twin ATC: Synthetic ATC Communications with LLMs**

## **Overview**

This repository is part of the Boeing Gen-Twin ATC research initiative, focused on simulating realistic air traffic control (ATC) communications and evaluating large language models (LLMs) fine-tuned on aviation documentation. 

Our goal is to improve airspace safety and efficiency by generating, fine-tuning, and evaluating models using domain-specific question answering (QA) tasks and synthetic pilot–controller dialogues.

---

Project Goals

- Simulate realistic **pilot–controller communications** using structured templates and placeholder tags.
- Fine-tune and evaluate the `microsoft/phi-4`, `meta/Llama 3.1 8B`, `deepseek-ai/DeepSeek 7B`,`google/Gemma 7B` model on FAA documents and ATC SOPs.
- Benchmark the model using a wide range of **NLP evaluation metrics**.
- Enable **digital twin simulations** for airspace safety, controller training and contextual model retraining.

---

# **Main Features**

-  Fine-tuning of `microsoft/phi-4`, `meta/Llama 3.1 8B`, `deepseek-ai/DeepSeek 7B`,`google/Gemma 7B` on aviation-related documents PDFs (FAA, SOPs, AIM, NOTAMs, etc.)
-  QA-based evaluation using metrics like:
  - Manual Accuracy
  - Cosine Similarity
  - BLEU, ROUGE-L, METEOR
  - BERTScore
  - Edit Distance
  - Perplexity
    
- **Synthetic ATC Communication Generation:**
  - Template-based prompts using aviation tags (e.g., `<FH>`, `<RWY>`, `<TF>`)
  - Pilot-initiated messages → ATC response prediction
  - Call sign recognition (ACC, WER) and NLP metric evaluation
  - Result logging to CSV for all QA evaluations and synthetic generations

---

## **Dataset Sources**

PDFs and QA pairs were derived from the following aviation documents:

- [x] FAA Pilot's Handbook of Aeronautical Knowledge
- [x] AIM (Aeronautical Information Manual)
- [x] (GSO) Greensboro Airport ATC Standard Operating Procedures
- [x] Departure Charts/Arrival Procedures (e.g., BLOCC TWO, QUAKER EIGHT)
- [x] Risk Management Handbook
- [x] IFR Alternate Minimums
- [x] Remote Pilot – sUAS Guide
- [x] Airport Master Plan Update (GSO)
- [x] Takeoff Minimums & Departure Procedures


QA Evaluation Results (Sample)

Document	                      Manual Acc	Cosine	BLEU	ROUGE-L	Edit Dist	Perplexity BertScore
IFR Alternate Airport Minimums	0.85	      0.91	  0.42	0.68	   3.1	     4.21      90.01


*All documents used for training and evaluation are publicly available through FAA and airport planning sources.*


---

Sythetic Communication Generation

Tags & Synthetic Template Types

**Structured tags used in 100 synthetic communications**:
- `<FH>`: Flight callsign
- `<ALT>`, `<NEW_ALT>`: Altitudes
- `<RWY>`: Runways
- `<TF>`, `<AF>`, `<GF>`: ATC Frequencies
- `<TXWY>`: Taxiways
- `<WIND>`: Weather/wind
- `<HDG>`: Heading
- `<DIST>`: Distance
- `<DEST>`: Destination
- `<ROUTE>`: Ground taxi routing

---

**Types of communications simulated**:

1. Pilot check-in with approach
2. Radar contact and descent instructions
3. Heading and altitude changes
4. Tower handoff
5. Final approach calls
6. Landing clearance
7. Ground contact and taxi
8. Parking and gate instructions

---


Synthetic ATC Generation (Sample)
Prompt:
Greensboro Tower, N123AB, ILS 23L, 8 miles out.

Model Response:
N123AB, wind 230 at 10, Runway 23L cleared to land.

Metrics:

Call Sign Accuracy: 0.99

CallWER: 0.01

BLEU: 0.82

ROUGE-L: 0.77

Edit Distance: 4

Perplexity: 3.5

---

**Model I/O Design**

Model Input: Structured prompt with context
Context: [REFERENCE ANSWER] \n\nQuestion: [QUESTION] \nAnswer:

Model Output: Predicted answer from LLM

Compared Against: Reference answer from dataset

**Synthetic Setup:**

Pilot message (input) → ATC message (generated output)
Used for call sign accuracy & semantic match evaluation


## **Repository Structure**


---

## **How to Run**

1. Install dependencies:
```bash
pip install -r requirements.txt




Synthetic ATC Communication:

Prompt: Greensboro Tower, N466C7, ILS 23L, 8 miles out.
Response: N466C7, wind 230 at 10, Runway 23L cleared to land.


Research Impact:

This project enables digital twin environments to simulate realistic ATC communication scenarios using Generative AI. The ability to generate and evaluate ATC conversations facilitates improved training, safety assessments, and automation integration.




Citation
If this repository or method supports your work, please cite:

@inprogress{

}


Acknowledgements:

Special thanks to:

Dr. Hamidreza Moradi and the Boeing Gen-Twin team

FAA and NASA for publicly accessible aviation materials

OpenAI, Google, Meta, Microsoft, and Hugging Face for model access and tooling



Contact

Stefan Green
Ph.D. Researcher, Computer Science

For questions or collaborations, contact: Stefan Green
email: smgreen1@aggies.ncat.edu
linkedin: [https://www.linkedin.com/in/stefanmgreen/]
