import os
import nltk
nltk.download('wordnet')
import csv
import torch
import time
import psutil
import editdistance
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoConfig
from nltk.translate.meteor_score import meteor_score

# ✅ **Set device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ **Load model and tokenizer**
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Lower memory use
    force_download=True
).to(device)

# ✅ **Load embedding model for Cosine Similarity**
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# ✅ **Initialize smoothing function for BLEU score**
smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Suppress Roberta pooler warning by disabling it
config = AutoConfig.from_pretrained("roberta-large")
config.update({"use_pooler": False})  # Disable pooler layer


# ✅ **Function to normalize text (handles encoding issues)**
def normalize_text(text):
    if text is None:
        return ""
    return text.lower().strip().replace("’", "'").replace("\x92", "'").replace("â\x80\x99", "'")


# ✅ **Generate response from LLaMA 3.1**
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=100,
                             temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ✅ **Calculate Perplexity**
def calculate_perplexity(prediction):
    with torch.no_grad():
        inputs = tokenizer(prediction, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        perplexity = torch.exp(loss).item()
        return perplexity


# QA pairs for all documents
qa_pairs = {
    "IFR Alternate Airport Minimums": {
        "questions": [
            "What are the criteria for using an alternate airport in IFR flight planning?",
            "Which approach operations are considered precision approaches?",
            "What are the alternate minima for Anderson Regional Airport?",
            "What is the restriction for Asheville Regional Airport when the control tower is closed?",
            "How is the category of aircraft determined for alternate minima?",
            "What is the standard alternate minima for non-precision approaches?",
            "What are the alternate minima requirements for helicopters?",
            "What does the symbol 'NA' indicate on the approach chart?",
            "What is the significance of 14 CFR 91.169 in alternate airport planning?",
            "What are the alternate minima for Rocky Mount Regional Airport?",
            "What does 'ILS or LOC' signify in alternate minima charts?",
            "How are RNAV (GPS) approaches classified in alternate planning?",
            "What is the restriction for RNAV approaches at Michael J. Smith Field?",
            "What are the considerations for Category D aircraft in alternate planning?",
            "What are the minima for Concord-Padgett Regional Airport?",
            "How are non-standard minima depicted on the charts?",
            "What are the minima for airports with no local weather reporting?",
            "What is the difference between standard and non-standard minima?",
            "What is the alternate minima for Columbia Metropolitan Airport?",
            "How does the FAA regulate alternate airport minima?"
        ],
        "answers": [
            ("Pilots must review the IFR Alternate Minimums Notes for suitability.", "Page 1"),
            ("ILS, PAR, and GLS operations are considered precision approaches.", "Page 1"),
            ("ILS or LOC Rwy 5: LOC, Category D, 800-2¼.", "Page 2"),
            ("Approach not allowed when the control tower is closed.", "Page 3"),
            ("Based on the aircraft's approach category (speed and performance).", "Page 1"),
            ("Standard for non-precision: 800-2.", "Page 1"),
            ("Helicopters require 200 feet above the published ceiling.", "Page 1"),
            ("'NA' indicates non-availability due to weather or facility issues.", "Page 1"),
            ("14 CFR 91.169 outlines alternate minima requirements.", "Page 1"),
            ("RNAV (GPS) Rwy 4: Category D, 800-2¼.", "Page 2"),
            ("ILS or LOC indicates precision landing guidance.", "Page 2"),
            ("RNAV (GPS) is considered non-precision.", "Page 1"),
            ("NA when local weather is unavailable.", "Page 2"),
            ("Category D minima vary based on airport facilities.", "Page 1"),
            ("RNAV (GPS) Rwy 2: Category D, 800-2¼.", "Page 2"),
            ("Non-standard minima are explicitly listed with restrictions.", "Page 1"),
            ("Minima are higher or NA when weather reporting is unavailable.", "Page 2"),
            ("Standard minima use 600-2 for precision, 800-2 for non-precision.", "Page 1"),
            ("Category D minima: ILS, 900-2¼; LOC, 900-2½.", "Page 3"),
            ("FAA regulations ensure minimum safety standards for alternates.", "Page 1")
        ]
    },
    "Aeronautical Chart Users’ Guide": {
        "questions": [
            "What is the purpose of the Aeronautical Chart User's Guide?",
            "What information is provided on a sectional chart?",
            "How are controlled airspaces depicted on charts?",
            "What symbols indicate special use airspace?",
            "What details are shown for airports on sectional charts?",
            "How is terrain elevation represented on the chart?",
            "What are the symbols for obstructions like towers?",
            "How are heliports represented on the chart?",
            "What is the depiction for restricted airspace?",
            "What information does the legend provide for flight planning?",
            "What is the significance of MEF on sectional charts?",
            "How is Class B airspace represented on sectional charts?",
            "What symbols represent runway lighting systems?",
            "How are communication frequencies indicated on the charts?",
            "What are VOR symbols on aeronautical charts?",
            "What is the role of the chart scale in navigation?",
            "How are temporary flight restrictions depicted?",
            "What is the symbol for military operations areas?",
            "What are the color codes used for elevation depiction?",
            "How are prohibited areas indicated?"
        ],
        "answers": [
            ("The guide provides pilots with chart symbology and interpretation.", "Page 1"),
            ("Sectional charts show airspace boundaries, topography, and navigation aids.", "Page 3"),
            ("Controlled airspaces are shown with solid or dashed blue and magenta lines.", "Page 4"),
            ("Special use airspaces are depicted with hatched lines or annotations.", "Page 5"),
            ("Airports are shown with symbols indicating type, size, and services available.", "Page 6"),
            ("Terrain elevation is depicted using contour lines and color gradients.", "Page 7"),
            ("Obstructions like towers are marked with a tower symbol and height data.", "Page 8"),
            ("Heliports are depicted with a capital H within a circle.", "Page 9"),
            ("Restricted airspace is outlined with blue-hatched lines.", "Page 10"),
            ("The legend explains chart symbology and abbreviations.", "Page 1"),
            ("MEF stands for Maximum Elevation Figures, providing clearance heights.", "Page 3"),
            ("Class B airspace is depicted with solid blue lines.", "Page 4"),
            ("Runway lighting systems are represented by specific symbols next to airport markers.", "Page 6"),
            ("Communication frequencies are listed near the associated airspace or navigation aid.", "Page 8"),
            ("VOR symbols include a hexagon with a dot in the center.", "Page 5"),
            ("The chart scale helps pilots measure distances accurately.", "Page 3"),
            ("Temporary flight restrictions are indicated with dashed red lines.", "Page 4"),
            ("Military operations areas are marked with magenta-hatched boundaries.", "Page 7"),
            ("Color codes for elevation range from green (low) to brown (high).", "Page 3"),
            ("Prohibited areas are depicted with blue-hatched lines and labels like 'P-123'.", "Page 4")
        ]
    },
    "Greensboro SOP": {
        "questions": [
            "What are the responsibilities of the tower controller at Greensboro Airport?",
            "How are runway operations managed during peak hours?",
            "What is the procedure for simultaneous landings on intersecting runways?",
            "What are the taxiway designations at Greensboro Airport?",
            "What is the standard communication protocol for clearance delivery?",
            "How is emergency landing prioritized in Greensboro SOP?",
            "What are the procedures for low visibility operations?",
            "What is the traffic flow pattern for RWY 23?",
            "What are the weather minima required for VFR operations at the airport?",
            "How are missed approaches handled at Greensboro Airport?",
            "What is the protocol for handling wake turbulence at the airport?",
            "How are temporary runway closures communicated to pilots?",
            "What are the training requirements for ATC personnel?",
            "How does the SOP address noise abatement procedures?",
            "What is the procedure for handling diversions to Greensboro Airport?",
            "What are the emergency frequencies used at the airport?",
            "How is runway incursion prevention managed?",
            "What are the signage requirements for taxiway intersections?",
            "What is the standard handoff procedure between ATC sectors?",
            "What are the lighting requirements for night operations?"
        ],
        "answers": [
            ("Tower controllers coordinate ground and air traffic movements.", "Page 2"),
            ("Runway usage alternates during peak hours based on traffic load.", "Page 4"),
            ("Simultaneous landings require strict coordination and timing.", "Page 5"),
            ("Taxiways are labeled A through K with clear signage.", "Page 3"),
            ("Clearance delivery uses standard phraseology and procedures.", "Page 7"),
            ("Emergency landings are given immediate priority and runway access.", "Page 6"),
            ("Low visibility operations involve Category II/III ILS procedures.", "Page 8"),
            ("Traffic flows north to south for RWY 23.", "Page 9"),
            ("VFR requires a ceiling of 1,000 feet and visibility of 3 miles.", "Page 10"),
            ("Missed approaches are rerouted to holding patterns.", "Page 5"),
            ("Wake turbulence is managed using staggered spacing.", "Page 11"),
            ("Runway closures are broadcast via NOTAMs.", "Page 12"),
            ("ATC personnel must complete annual training and certifications.", "Page 13"),
            ("Noise abatement includes specific approach and departure routes.", "Page 14"),
            ("Diversions are managed with pre-allocated holding areas.", "Page 15"),
            ("Emergency frequencies include 121.5 MHz and 243.0 MHz.", "Page 16"),
            ("Runway incursions are prevented using ASDE-X systems.", "Page 17"),
            ("Taxiway intersections have mandatory stop bar lights.", "Page 18"),
            ("Sector handoffs involve notifying adjacent sectors in advance.", "Page 19"),
            ("Lighting includes runway edge lights and PAPI systems.", "Page 20")
        ]
    },
    "GreensboroATC Standard Operating Procedures": {
        "questions": [
            "What is the role of the Greensboro TRACON in air traffic management?",
            "How are departure clearances issued in the ATC SOP?",
            "What are the frequency allocations for the Greensboro ATC sectors?",
            "How is radar separation maintained between aircraft?",
            "What is the procedure for issuing holding instructions?",
            "What are the guidelines for handling lost communication with an aircraft?",
            "How does Greensboro ATC coordinate with adjacent facilities?",
            "What is the protocol for runway changes during active operations?",
            "What is the role of the Local Control position at the airport?",
            "How are wake turbulence separation minima enforced?",
            "What is the standard spacing for arrivals on the ILS approach?",
            "What are the VFR flight following procedures in the ATC SOP?",
            "What is the procedure for handling emergencies in the ATC SOP?",
            "How are weather deviations handled in the Greensboro ATC area?",
            "What is the coordination protocol for military operations?",
            "How is non-standard traffic handled in the ATC SOP?",
            "What are the handoff procedures for TRACON to Tower control?",
            "What are the phraseology requirements for issuing clearances?",
            "What is the altitude assignment policy for departing aircraft?",
            "How are special VFR clearances issued in the ATC SOP?"
        ],
        "answers": [
            ("TRACON handles arrivals and departures within Class B and surrounding airspace.", "Page 1"),
            ("Departure clearances are issued via pre-defined standard routes.", "Page 3"),
            ("Frequency allocations are specific to individual sectors and altitudes.", "Page 2"),
            ("Radar separation is maintained at a minimum of 3 miles.", "Page 4"),
            ("Holding instructions include holding fixes and EFC times.", "Page 5"),
            ("Lost communication procedures involve squawking 7600 and proceeding under VFR.", "Page 6"),
            ("Coordination includes pre-allocated routes for handoffs.", "Page 7"),
            ("Runway changes require tower and ground coordination.", "Page 8"),
            ("Local Control manages active runways and immediate departures.", "Page 9"),
            ("Wake turbulence is managed using staggered spacing and category weights.", "Page 10"),
            ("ILS approaches require 5-mile spacing for sequential arrivals.", "Page 11"),
            ("Flight following is initiated on request and monitored by ATC.", "Page 12"),
            ("Emergencies involve declaring a state and activating airport response.", "Page 13"),
            ("Weather deviations are granted immediate approval for safety.", "Page 14"),
            ("Military coordination involves pre-planned schedules and priorities.", "Page 15"),
            ("Non-standard traffic requires coordination and direct routing.", "Page 16"),
            ("TRACON transfers to Tower at 5 miles from the airport.", "Page 17"),
            ("Clearance phraseology follows FAA standards and local variations.", "Page 18"),
            ("Altitude is assigned based on route and performance capabilities.", "Page 19"),
            ("Special VFR is granted under specific weather minima.", "Page 20")
        ]
    },
    "GSO Airport Master Plan Update": {
        "questions": [
            "What are the key objectives outlined in the GSO Airport Master Plan?",
            "What is the projected passenger growth for the next 10 years?",
            "What improvements are planned for the terminal facilities?",
            "What are the proposed changes to runway configurations?",
            "How is the cargo traffic expected to evolve in the future?",
            "What environmental considerations are addressed in the master plan?",
            "What is the proposed timeline for the expansion of Taxiway Bravo?",
            "What are the budget estimates for the master plan's implementation?",
            "What are the critical infrastructure projects outlined in the plan?",
            "How is the airport planning to improve access to ground transportation?",
            "What safety enhancements are proposed for the airfield?",
            "How does the master plan address noise abatement measures?",
            "What are the plans for general aviation development?",
            "What community engagement strategies are outlined in the master plan?",
            "What new technologies are proposed for air traffic management?",
            "What are the plans for expanding airport parking facilities?",
            "What strategies are proposed for enhancing airport revenue streams?",
            "What upgrades are planned for the airport's IT infrastructure?",
            "What partnerships are highlighted for the plan's success?",
            "What are the primary challenges identified in the master plan?"
        ],
        "answers": [
            ("The objectives include capacity enhancement and infrastructure modernization.", "Page 3"),
            ("Passenger growth is projected to increase by 25% in the next decade.", "Page 5"),
            ("Terminal upgrades include additional gates and improved passenger flow.", "Page 7"),
            ("Runway extensions and reconfigurations are planned for increased capacity.", "Page 9"),
            ("Cargo traffic is expected to grow by 15% annually.", "Page 11"),
            ("Environmental impact assessments and mitigation measures are included.", "Page 13"),
            ("The expansion of Taxiway Bravo is planned over the next 3 years.", "Page 15"),
            ("The budget for the master plan implementation is estimated at $1 billion.", "Page 17"),
            ("Critical projects include runway expansions and terminal upgrades.", "Page 19"),
            ("Ground transportation access will be improved with new roadways.", "Page 21"),
            ("Safety enhancements include upgraded lighting and signage.", "Page 23"),
            ("Noise abatement measures involve revised flight paths.", "Page 25"),
            ("General aviation development includes new hangars and ramps.", "Page 27"),
            ("Community engagement involves public consultations and workshops.", "Page 29"),
            ("Proposed technologies include enhanced radar systems.", "Page 31"),
            ("Airport parking facilities will be expanded by 50%.", "Page 33"),
            ("Strategies include attracting more airlines and increasing retail options.", "Page 35"),
            ("IT upgrades include advanced security and operational systems.", "Page 37"),
            ("Partnerships with local governments and airlines are emphasized.", "Page 39"),
            ("Challenges include funding constraints and regulatory approvals.", "Page 41")
        ]
    },
    "Pilot’s Handbook of Aeronautical Knowledge": {
        "questions": [
            "What are the fundamental principles of flight?",
            "What is the role of the ailerons during flight?",
            "How does the center of gravity affect aircraft performance?",
            "What are the different types of drag experienced by an aircraft?",
            "How is lift generated according to the handbook?",
            "What is the function of the horizontal stabilizer?",
            "What is the relationship between angle of attack and lift?",
            "What are the key components of the aircraft's control system?",
            "How is engine performance affected by altitude?",
            "What is the significance of the four forces of flight?",
            "What are the factors affecting stall speed?",
            "How is stability maintained in flight?",
            "What is the purpose of trim tabs on the aircraft?",
            "How does weight distribution affect takeoff performance?",
            "What are the safety measures for preventing spatial disorientation?",
            "What is the standard approach procedure for landing?",
            "How does the handbook describe adverse yaw?",
            "What are the recommended weather minima for VFR flights?",
            "What is the significance of ground effect during takeoff and landing?",
            "How is turbulence categorized in the handbook?"
        ],
        "answers": [
            ("Flight is based on the principles of lift, thrust, drag, and weight.", "Page 1"),
            ("Ailerons control roll and maintain lateral stability.", "Page 3"),
            ("The center of gravity affects stability and controllability.", "Page 5"),
            ("Drag is categorized into parasitic and induced drag.", "Page 7"),
            ("Lift is generated by the pressure differential over the wing surfaces.", "Page 9"),
            ("The horizontal stabilizer maintains longitudinal stability.", "Page 11"),
            ("An increase in angle of attack increases lift until the critical angle.", "Page 13"),
            ("The control system includes ailerons, rudder, and elevator.", "Page 15"),
            ("Engine performance decreases with altitude due to reduced air density.", "Page 17"),
            ("The four forces of flight are lift, thrust, drag, and weight.", "Page 19"),
            ("Stall speed increases with weight and load factor.", "Page 21"),
            ("Stability is maintained through the aircraft's aerodynamic design.", "Page 23"),
            ("Trim tabs help reduce the pilot's control forces.", "Page 25"),
            ("Weight distribution affects the aircraft's center of gravity.", "Page 27"),
            ("Spatial disorientation is prevented through proper training and instruments.", "Page 29"),
            ("The standard approach involves a stabilized descent and alignment.", "Page 31"),
            ("Adverse yaw is caused by differential drag during turns.", "Page 33"),
            ("VFR flights require a minimum ceiling of 1,000 feet.", "Page 35"),
            ("Ground effect reduces induced drag near the ground.", "Page 37"),
            ("Turbulence is categorized into light, moderate, and severe.", "Page 39")
        ]
    },
    "Remote Pilot Small Unmanned Aircraft Systems Study Guide": {
        "questions": [
            "What are the eligibility requirements for obtaining a remote pilot certificate?",
            "What is the role of the FAA in regulating small unmanned aircraft systems (sUAS)?",
            "What are the limitations for sUAS operations under Part 107?",
            "How is controlled airspace defined for sUAS operations?",
            "What are the weather requirements for operating a small unmanned aircraft?",
            "What is the maximum allowable altitude for sUAS operations?",
            "What are the requirements for visual line of sight during sUAS operations?",
            "How are remote pilots expected to handle emergency situations?",
            "What are the key components of a preflight inspection for sUAS?",
            "What is the process for reporting sUAS accidents to the FAA?",
            "What are the roles and responsibilities of the remote pilot in command (RPIC)?",
            "What is the significance of NOTAMs for sUAS operations?",
            "What are the battery maintenance guidelines for small unmanned aircraft?",
            "How does the guide define operations over people?",
            "What are the specific requirements for night operations under Part 107?",
            "How is airspace authorization obtained for operations in Class B, C, D, or E airspace?",
            "What is the role of aeronautical charts in planning sUAS operations?",
            "What are the requirements for recurrent testing for remote pilots?",
            "What is the significance of the small unmanned aircraft registration system?",
            "What are the rules for operating sUAS over moving vehicles?"
        ],
        "answers": [
            ("Applicants must be at least 16 years old and pass an FAA knowledge test.", "Page 1"),
            ("The FAA regulates sUAS operations to ensure airspace safety.", "Page 3"),
            ("Operations must remain within 400 feet of the ground and during daylight hours.", "Page 5"),
            ("Controlled airspace includes Class B, C, D, and certain parts of Class E.", "Page 7"),
            ("Weather must include visibility of 3 miles and cloud clearance of 500 feet.", "Page 9"),
            ("Maximum altitude is 400 feet above ground level unless within 400 feet of a structure.", "Page 11"),
            ("Remote pilots must maintain visual contact with the sUAS at all times.", "Page 13"),
            ("Emergencies must be handled to prioritize safety and reported to the FAA.", "Page 15"),
            ("Preflight checks include inspecting batteries, propellers, and the flight control system.", "Page 17"),
            ("Accidents must be reported within 10 days if they result in serious injury or property damage.",
             "Page 19"),
            ("The RPIC is responsible for the safe operation of the sUAS.", "Page 21"),
            ("NOTAMs provide crucial information about airspace restrictions.", "Page 23"),
            ("Batteries must be stored and charged according to the manufacturer’s instructions.", "Page 25"),
            ("Operations over people require compliance with strict safety measures.", "Page 27"),
            ("Night operations require anti-collision lighting visible for 3 miles.", "Page 29"),
            ("Airspace authorization is obtained through the FAA's LAANC system.", "Page 31"),
            ("Aeronautical charts help identify airspace classes and restrictions.", "Page 33"),
            ("Recurrent testing is required every 24 months.", "Page 35"),
            ("All sUAS must be registered if they weigh more than 0.55 pounds.", "Page 37"),
            ("Operations over moving vehicles are prohibited unless the vehicle is in a restricted area.", "Page 39")
        ]
    },
    "Risk Management Handbook": {
        "questions": [
            "What is the purpose of the Risk Management Handbook?",
            "What is the risk management process described in the handbook?",
            "How does the handbook define 'hazard' in aviation?",
            "What are the five steps in the risk management decision-making process?",
            "How are risk assessments conducted for flight operations?",
            "What is the PAVE checklist, and how is it used in risk management?",
            "What is the 5P model, and how does it apply to aviation safety?",
            "What are the key elements of situational awareness in risk management?",
            "How does the handbook address human factors in aviation safety?",
            "What is the IMSAFE checklist, and how is it used by pilots?",
            "What are the guidelines for identifying and mitigating risks in flight planning?",
            "How is risk tolerance defined in the handbook?",
            "What are the tools recommended for ongoing risk monitoring during flight?",
            "How does the handbook define acceptable levels of risk?",
            "What are the decision-making models described in the handbook?",
            "What is the significance of aeronautical decision-making (ADM) in risk management?",
            "What are the strategies for managing in-flight emergencies?",
            "How does the handbook address weather-related risks?",
            "What is the role of post-flight debriefing in risk management?",
            "How does the handbook recommend incorporating technology into risk management?"
        ],
        "answers": [
            ("The handbook provides guidelines for identifying and mitigating aviation risks.", "Page 2"),
            ("The risk management process involves identifying, analyzing, and mitigating risks.", "Page 4"),
            ("A hazard is any condition that could lead to an accident.", "Page 6"),
            ("The steps include identifying hazards, assessing risks, and implementing controls.", "Page 8"),
            ("Risk assessments are conducted using qualitative and quantitative methods.", "Page 10"),
            ("The PAVE checklist stands for Pilot, Aircraft, Environment, and External Pressures.", "Page 12"),
            ("The 5P model includes Plan, Plane, Pilot, Passengers, and Programming.", "Page 14"),
            ("Situational awareness involves understanding the environment and predicting outcomes.", "Page 16"),
            ("Human factors include fatigue, stress, and decision-making errors.", "Page 18"),
            ("The IMSAFE checklist evaluates Illness, Medication, Stress, Alcohol, Fatigue, and Emotion.", "Page 20"),
            ("Risk identification involves analyzing weather, aircraft performance, and route conditions.", "Page 22"),
            ("Risk tolerance is the level of risk deemed acceptable for a specific operation.", "Page 24"),
            ("Monitoring tools include checklists, instruments, and pilot reports.", "Page 26"),
            ("Acceptable risk levels are those that maintain safety and operational goals.", "Page 28"),
            ("Models include the DECIDE model and the 3P model.", "Page 30"),
            ("ADM involves systematic decision-making to enhance safety.", "Page 32"),
            ("Strategies include prioritizing tasks and maintaining communication.", "Page 34"),
            ("Weather risks involve turbulence, visibility, and wind conditions.", "Page 36"),
            ("Debriefing identifies lessons learned and areas for improvement.", "Page 38"),
            ("Technology includes automation, GPS, and weather forecasting tools.", "Page 40")
        ]
    },
    "TAKEOFF MINIMUMS (OBSTACLE) DEPARTURE PROCEDURES": {
        "questions": [
            "What are the standard takeoff minimums for commercial operators?",
            "How are obstacle departure procedures defined?",
            "What are the visibility requirements for takeoff under IFR conditions?",
            "What is the purpose of a Diverse Vector Area (DVA)?",
            "How are non-standard takeoff minimums communicated to pilots?",
            "What is the role of climb gradients in departure planning?",
            "What is the significance of runway lighting systems for IFR departures?",
            "How does the handbook address takeoff in low visibility conditions?",
            "What are the criteria for designating a takeoff alternate airport?",
            "What is the purpose of the Obstacle Identification Surface (OIS)?",
            "How are diverse departure procedures implemented at airports?",
            "What are the minimum climb gradients for obstacle clearance?",
            "What are the requirements for pilots when using visual climb over airport (VCOA)?",
            "How are departure procedures published in airport charts?",
            "What is the process for obtaining clearance for a non-standard departure?",
            "What are the procedures for handling engine failure during takeoff?",
            "What is the importance of NOTAMs in takeoff planning?",
            "What are the minimum weather requirements for takeoff at uncontrolled airports?",
            "What is the role of TERPS in defining obstacle clearance?",
            "How are mountainous terrain departures managed?"
        ],
        "answers": [
            ("Standard minimums include 1 mile visibility for single-engine aircraft.", "Page 1"),
            ("ODPs provide instructions for safe departure considering terrain and obstacles.", "Page 3"),
            ("IFR takeoff requires a minimum of 1/2 mile visibility.", "Page 5"),
            ("DVAs ensure safe vectoring by ATC in areas with no published procedures.", "Page 7"),
            ("Non-standard minimums are listed in terminal procedure publications.", "Page 9"),
            ("Climb gradients ensure obstacle clearance during initial climb.", "Page 11"),
            ("Lighting systems include approach lights and runway edge lights.", "Page 13"),
            ("Low visibility takeoff uses RVR readings and specific lighting systems.", "Page 15"),
            ("A takeoff alternate is required when departure airport weather is below minimums.", "Page 17"),
            ("OIS defines the obstacle-free airspace during departure.", "Page 19"),
            ("Diverse procedures allow departures in all directions if no obstacles exist.", "Page 21"),
            ("Minimum gradients are specified in feet per nautical mile.", "Page 23"),
            ("VCOA requires pilots to maintain visibility and specific altitudes.", "Page 25"),
            ("Departure procedures are published in the airport's approach charts.", "Page 27"),
            ("Non-standard clearances are coordinated with ATC.", "Page 29"),
            ("Engine-out procedures involve returning to the airport or a safe landing area.", "Page 31"),
            ("NOTAMs provide updates on changes to departure procedures.", "Page 33"),
            ("Minimums at uncontrolled airports depend on terrain and local regulations.", "Page 35"),
            ("TERPS define obstacle clearance standards for instrument procedures.", "Page 37"),
            ("Mountainous departures involve increased climb gradients and route planning.", "Page 39")
        ]
    }
}


# ✅ **Evaluate Model**
accuracy_results = {}
start_time = time.time()

for doc_name, qa_data in qa_pairs.items():
    print(f"Processing document: {doc_name}")

    manual_scores, cosine_scores, bleu_scores, rouge_scores = [], [], [], []
    edit_distances, perplexities, bert_scores, meteor_scores = [], [], [], []

    for question, (reference, _) in zip(qa_data["questions"], qa_data["answers"]):
        prediction = generate_answer(question)

        # **Manual Accuracy**
        manual_acc = 1 if reference.lower() in prediction.lower() else 0

        # **Cosine Similarity**
        ref_embed = embedding_model.encode(reference)
        pred_embed = embedding_model.encode(prediction)
        cos_sim = cosine_similarity([ref_embed], [pred_embed])[0][0]

        # **BLEU Score**
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)

        # **ROUGE-L Score**
        rouge_score_val = scorer.score(reference, prediction)['rougeL'].fmeasure

        # **Edit Distance**
        edit_dist = editdistance.eval(reference.split(), prediction.split())

        # **Perplexity**
        perplexity = calculate_perplexity(prediction)

        # # **BERTScore**
        # P, R, F1 = bert_score([prediction], [reference], lang="en")
        # bert_f1 = F1.mean().item()
        # **BERTScore**
        P, R, F1 = bert_score([prediction], [reference], lang="en", model_type="roberta-large")
        bert_f1 = F1.mean().item()

        # **METEOR Score**
        # METEOR Score (FIXED)
        meteor = meteor_score([reference.split()], prediction.split())

        # **Append scores**
        manual_scores.append(manual_acc)
        cosine_scores.append(cos_sim)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_score_val)
        edit_distances.append(edit_dist)
        perplexities.append(perplexity)
        bert_scores.append(bert_f1)
        meteor_scores.append(meteor)

        # **Print Debug Info**
        print(f"\nQuestion: {question}")
        print(f"Reference: {reference}")
        print(f"Predicted: {prediction}")
        print(f"Manual Accuracy: {manual_acc}")
        print(f"Cosine Similarity: {cos_sim:.2f}")
        print(f"BLEU Score: {bleu:.2f}")
        print(f"ROUGE-L Score: {rouge_score_val:.2f}")
        print(f"Edit Distance: {edit_dist}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"BERTScore F1: {bert_f1:.2f}")
        print(f"METEOR Score: {meteor:.2f}")

    # **Store Average Results**
    accuracy_results[doc_name] = {
        "manual_accuracy": sum(manual_scores) / len(manual_scores),
        "cosine_similarity": sum(cosine_scores) / len(cosine_scores),
        "bleu_score": sum(bleu_scores) / len(bleu_scores),
        "rougeL_score": sum(rouge_scores) / len(rouge_scores),
        "avg_edit_distance": sum(edit_distances) / len(edit_distances),
        "avg_perplexity": sum(perplexities) / len(perplexities),
        "bert_score_f1": sum(bert_scores) / len(bert_scores),
        "meteor_score": sum(meteor_scores) / len(meteor_scores)
    }

# ✅ **Save Metrics to CSV**
csv_file = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/llama_metrics_pre_finetune_bertandmet_newquery.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Write headers
    writer.writerow(["model_name", "document", "manual_accuracy", "cosine_similarity", "bleu_score",
                     "rougeL_score", "avg_edit_distance", "avg_perplexity", "bert_score_f1", "meteor_score"])

    # Write data
    for doc_name, metrics in accuracy_results.items():
        writer.writerow([
            model_name,
            doc_name,
            metrics["manual_accuracy"],
            metrics["cosine_similarity"],
            metrics["bleu_score"],
            metrics["rougeL_score"],
            metrics["avg_edit_distance"],
            metrics["avg_perplexity"],
            metrics["bert_score_f1"],
            metrics["meteor_score"]
        ])

print(f"\nMetrics saved to {csv_file}")

exec_time = time.time() - start_time
print(f"\nTotal Execution Time: {exec_time / 60:.2f} minutes")
print(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
print(f"Final CPU Memory: {psutil.virtual_memory().used / 1e9:.2f}GB")
