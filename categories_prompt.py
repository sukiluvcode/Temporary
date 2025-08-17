from langchain_openai import ChatOpenAI


text = """The HEA with a nominal composition of V10Cr15Mn5Fe35Co10Ni25 (at%) was fabricated using vacuum induction melting furnace using pure elements of V, Cr, Mn, Fe, Co, and Ni (purity >99.9%). The as-cast sample was subjected to homogenization heat treatment at 1100 °C for 6 h under an Ar atmosphere, followed by water quenching. The homogenized sample was cold rolled through multiple passes with a final rolling reduction ratio of ≈79% (from 6.2 to 1.3 mm). The disk-shaped samples (10 mm diameter) were prepared from the cold rolled sheet using electro-discharge machining. The disk samples were annealed at two different conditions (900 °C for 10 min and 1100 °C for 60 min) to obtain microstructure with fine grains and coarse grains, respectively. Finally, the HPT process was carried out on the annealed samples at different turns (N = 1/4, 1, and 5) using a pressure of 6 GPa and a rotation rate of 1 revolution per minute (rpm)."""
prompt = f"""You are reading experimental section of high entropy alloys (HEAs) paper and based on the description, you need to give you judgement about whether the author distinctly identified each material(s) frabricated
Here are some guidances you need to follow
- 1. If the author describes one processing route which produce only one sample, you give True as long as the chemical composition is given, else False.
- 2. If the author describes one processing route which produce serval samples with different chemical compositions, you give True if every sample is given with composition/symbol, else Flase. General formula like MnxCoCrNi is not acceptable.
- 3. If the author describes multiple processing routes (cases are i. with different fabrication method. ii. with different processing parameters, like temperature or duration etc. iii. any combination of i and ii.), and these routes produce multiple samples which have same composition, you give True if every of the samples are clearly labeled explicitly, else False.
- 4. If the author describes multiple processing routes and these route produce samples with different composition, you give True if every sample is given with composition/symbol, else Flase. General formula like MnxCoCrNi is not acceptable.
**symbol** means the author need to give explicitly defined symbol to identified the material, like A0, HEA-1, annealed-700 etc, but not some descriptional words.

Experimental section
{text}
"""

p = f"""You are analyzing the experimental section of a high entropy alloys (HEAs) research paper. Your task is to determine whether the author has distinctly identified each fabricated material.

**Task**: Return True if ALL materials are distinctly identified, False otherwise.

**Material Identification Requirements**:
- **Chemical composition**: Specific elemental ratios (e.g., "Mn0.2CoCrNi", "Al0.3CoCrFeNi0.5")
  - General formulas like "MnxCoCrNi" are NOT acceptable
- **Explicit symbol/label**: Clear identifier like "A0", "HEA-1", "Sample-700C", "as-cast"
  - Descriptive phrases without explicit labels are NOT acceptable

**Evaluation Rules**:

**Case 1 - Single processing route, single sample**:
- True: If chemical composition is provided
- False: If composition is missing

**Case 2 - Single processing route, multiple samples with different compositions**:
- True: If every sample has either specific composition OR explicit symbol/label
- False: If any sample lacks both composition and symbol/label

**Case 3 - Multiple processing routes, same composition**:
(Different methods, parameters, temperatures, durations, etc.)
- True: If every sample is explicitly labeled with unique identifiers
- False: If any sample lacks clear labeling

**Case 4 - Multiple processing routes, different compositions**:
- True: If every sample has either specific composition OR explicit symbol/label
- False: If any sample lacks both composition and symbol/label

**Experimental Section**:
{text}

**Response**: True or False"""


model = ChatOpenAI(model='gpt-4.1-mini', temperature=0.0)

print(model.invoke(prompt))

