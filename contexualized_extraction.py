# Implementation of contexualized extraction: Frist extract properties separately then link it with correponds synthesis routes via description of that material.

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from typing import Optional
from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import syn_template
from prompt import *

# suppose paragraphs are correctly labeled
# First we need to categorize the paper based on whether the material are distinctly identified
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0.0))
class SynCategorization(dspy.Signature):
    """system prompt"""
    text: str = dspy.InputField(description="The experimental section of a high entropy alloys (HEAs) research paper.")
    output: bool = dspy.OutputField()
SynCategorization.__doc__ = CATEGORIZE_DSPY
categorize_agent = dspy.ChainOfThought(SynCategorization)

# Define extraction agent for each property, here we test for (ys, uts, strain) & phase
class Material(BaseModel):
    composition: Optional[str] = Field(description="Chemical composition of the material, e.g., 'Mn0.2CoCrNi'")

class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="Chemical composition of the material, e.g., 'Mn0.2CoCrNi'")
    description: Optional[str] = Field(description="Description of the material. Give the description based on the processing method, e.g., 'as-cast', 'annealed at 900C'. Do not contain test condition which describes the testing setup rather than the material itself, such as tested under 700C, under salted environment. If material is given with composition only, without any description, return None.")
    refered: bool = Field(description="Indicate whether the material data is cited from other publications for comparison purposes.")


class StrengthTestBase(BaseModel):
    """Tensile/Compressive test results"""
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain with unit")
    temperature: Optional[str] = Field(description="Test temperature with unit")
    strain_rate: Optional[str] = Field(description="Strain rate with unit")
    other_test_conditions: str = Field(description="Other test conditions, like in salt, hydrogen charging, etc.")

class PhaseInfo(BaseModel):
    """Phase information"""
    phases: Optional[str] = Field(description="List of phases present in the material")

def build_result_model(name: str, doc:str, *bases):
    r = create_model(name, __base__=bases, __doc__=doc)
    return create_model('Records', records=(Optional[list[r]], ...))

Strength = build_result_model("Strength", "Strength test results with material description", MaterialDescriptionBase, StrengthTestBase)
Phase = build_result_model("Phase", "Phase information with material description", MaterialDescriptionBase, PhaseInfo)

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
strength_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_PROPERTY_SYS_GENERIC_PROMPT),
        ("user", STRENGTH_PROMPT),
    ]
)
phase_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_PROPERTY_SYS_GENERIC_PROMPT),
        ("user", PHASE_PROMPT),
    ]
)
strength_extraction_agent = strength_extraction_prompt | model.with_structured_output(Strength)
phase_extraction_agent = phase_extraction_prompt | model.with_structured_output(Phase)

# Define process and extract it
class Processes(Material):
    """Processing route for a material"""
    processes: str = Field(description="List of processing steps in chronological order, for each as a python dictionary. For example: [{'induction melting': {'temperature': 1500}}, {'annealed': {'temperature': 800, 'duration': '1h'}}]")
    
    

processes_format_dict = {k: v for k, v in syn_template.__dict__.items() if not k.startswith('__') and not callable(v)}
def format_processes(processes: list[str]) -> str:
    format_string = ""
    for process in processes:
        if process not in processes_format_dict:
            continue
        format_string += f"{processes_format_dict[process]}\n"
    return format_string.strip()
process_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_PROCESS_SYS_GENERIC_PROMPT),
        ("user", PROCESS_PROMPT),
    ]
)
process_extraction_agent = process_extraction_prompt | model.with_structured_output(Processes)




# Build a dict to collect material and correspond processing routes
material_process_d = {} # composed of extracted, e.g. (composition, description) as key, and processes as value
def extract_process(property: MaterialDescriptionBase, processing_format, material_process_dict, syn_context: str):
    composition, description, refered = property.composition, property.description, property.refered
    if refered:
        # this is a reference, we do not extract processing route
        return
    if not any([composition, description]):
        # this is not a valid extraction
        return
    if t:=(composition, description) in material_process_dict:
        return material_process_dict[t]
    # find same reference if possible, based on composition, description
    key_index = find_same_reference(composition, description, material_process_dict)
    if key_index is not None:
        process = material_process_dict[list(material_process_dict.keys())[key_index]]
        return process
    # otherwise, extract processing route
    process = process_extraction_agent.invoke(
        {
            "material_description": f"{composition}: {description}",
            "process_format": processing_format,
            "text": syn_context
        }
    )
    return process

def print_comp_description(material_process_dict):
    table = "| Index | Composition | Description |\n|-------|-------------|-------------|\n"
    for idx, ((composition, description), _) in enumerate(material_process_dict.items()):
        table += f"| {idx} | {composition or ''} | {description or ''} |\n"
    return table

class FindSameReference(dspy.Signature):
    """Find which row in the table refered to the same material as provided composition and description.
    Be very strict, only return the index if you are sure, otherwise return None."""
    table: str = dspy.InputField(description="Table of material composition and description, in markdown format")
    composition: str = dspy.InputField(description="The composition of the material, e.g., 'Mn0.2CoCrNi'")
    description: str = dspy.InputField(description="The description of the material, e.g., 'as-cast', 'annealed at 900C'")
    index: Optional[int] = dspy.OutputField(description="The index of the row")
get_same_reference_agent = dspy.ChainOfThought(FindSameReference)

def find_same_reference(composition: str, description: str, material_process_dict):
    if len(material_process_dict) == 0:
        return None
    index = get_same_reference_agent(
        table=print_comp_description(material_process_dict),
        composition=composition,
        description=description
    )
    return index
    
test_text = """
The HEA with a nominal composition of V10Cr15Mn5Fe35Co10Ni25 (at%) was fabricated using vacuum induction melting furnace using pure elements of V, Cr, Mn, Fe, Co, and Ni (purity >99.9%). The as-cast sample was subjected to homogenization heat treatment at 1100 °C for 6 h under an Ar atmosphere, followed by water quenching. The homogenized sample was cold rolled through multiple passes with a final rolling reduction ratio of ≈79% (from 6.2 to 1.3 mm). The disk-shaped samples (10 mm diameter) were prepared from the cold rolled sheet using electro-discharge machining. The disk samples were annealed at two different conditions (900 °C for 10 min and 1100 °C for 60 min) to obtain microstructure with fine grains and coarse grains, respectively. Finally, the HPT process was carried out on the annealed samples at different turns (N = 1/4, 1, and 5) using a pressure of 6 GPa and a rotation rate of 1 revolution per minute (rpm)label.
"""

print(extract_process(MaterialDescriptionBase(
        composition="V10Cr15Mn5Fe35Co10Ni25",
        description="CG sample after HPT processing (N=5)",
        refered=False
    ),
    format_processes(["induction_melting", "homogenized", "quenching", "cold_rolled", "annealed", "high_pressure_torsion"]),
    material_process_d,
    test_text
)
)

from langchain_core.runnables import RunnableSequence
def extract_property(input_, agent: RunnableSequence):
     return agent.invoke(input_).records

def extract_contextualized_main(agents_d: dict[str, list[RunnableSequence]]):
    # prepare texts
    paragraphs = ""
    property_text_d = prepare_property_text(paragraphs, types="")
    # ...
    # parallelized
    results = []
    for prop, agent in agents_d.items():
        text = property_text_d[prop]
        if not text:
            continue
        results_prop = extract_property({'text': text}, agent)
        if results_prop:
            results.extend(results_prop)
    # extract processes
    extract_process()
    

def prepare_property_text(paragraphs, types: list[str]) -> dict:
    # merge relevant text
    pass
    