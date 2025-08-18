CATEGORIZE_DSPY = """You are analyzing the experimental section of a high entropy alloys (HEAs) research paper. Your task is to determine whether the author has distinctly identified each fabricated material.

**Task**: Return True if ALL materials are distinctly identified, False otherwise.

**Material Identification Requirements**:
- **Chemical composition**: Specific elemental ratios (e.g., "Mn0.2CoCrNi", "Al0.3CoCrFeNi0.5")
  - General formulas like "MnxCoCrNi" are NOT acceptable
- **Explicit symbol**: Clear identifier like "A0", "HEA-1", "Sample-700C"
  - Descriptive phrases without explicit symbols are NOT acceptable

**Evaluation Rules**:

**Case 1 - Single processing route, single sample**:
- True: If chemical composition is provided
- False: If composition is missing

**Case 2 - Single processing route, multiple samples with different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol

**Case 3 - Multiple processing routes, same composition**:
(Different methods, parameters, temperatures, durations, etc.)
- True: If every sample is explicitly labeled with unique identifiers
- False: If any sample lacks clear labeling

**Case 4 - Multiple processing routes, different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol"""

EXTRACT_PROPERTY_SYS_GENERIC_PROMPT = """You are an expert in extracting structured material properties from scientific texts. Your task is to extract the following properties from the provided text, which should includes all relevant information, even if it is used for comparison purposes."""

STRENGTH_PROMPT = """Extract the mechanical property relevant to ys, uts and strain from the text

Follow these rules:
- Material composition should be in the form of chemical formula, e.g., "Mn0.2CoCrNi", not any descriptive phrases.
- If the value provided is a range, for example, "from 200 MPa to 300 MPa", extract it as "200-300 MPa".
- If the value is given as "greater than" or "less than", for example, "greater than 400 MPa", extract it as ">400 MPa".
- If the value is given as "approximately" or "around", for example, "approximately 250 MPa", extract it as "≈250 MPa".
- Otherwise, extract the value as it is.

text
{text}
"""

PHASE_PROMPT = """Extract the phase information from the text. Material composition should be in the form of chemical formula, e.g., "Mn0.2CoCrNi", not any descriptive phrases.

Here are some common phase types:
FCC, BCC, HCP, B2, intermetallic compounds (e.g., TiNi, Ti₂Ni, γ' precipitates, silicides, aluminides, sigma (σ) phases), carbides (e.g., WC), oxides (e.g., SiO₂), amorphous phases.
Guideline for phase extraction:
- If the author mentions ordered/disordered, include it in the phase information.
- Main phase should be listed first, followed by secondary phases, and so on.

text
{text}
"""

EXTRACT_PROCESS_SYS_GENERIC_PROMPT = """You are an expert in extracting processing routes for materials from scientific texts. Your task is to extract the processing route for the specified material."""

PROCESS_PROMPT = """Extract the processing route for below material from provided experimental section.
**{material_description}**

Guidance:
- Note that the sample I give probably be one of many samples synthesised in the experimental section. You only need to extract the processing route for the specified sample.

And follow below format rules, may some of them are optional depending on the process:
{process_format}

Experimental section
{text}
"""