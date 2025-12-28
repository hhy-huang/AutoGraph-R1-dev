REAFINER_JUDGEMENT_SYSTEM_PROMPT = """
As an advanced judgement assistant, your task is to judge whether the given question is answerable based on the provided KG context.

Evaluate whether the given question is answerable based on the provided KG context. Output your judgment in the following format:
<judge>Yes</judge> or <judge>No</judge>

**Important:** You must think carefully about the question and the KG context before making your judgment. And output your judgment result directly in the specified format.
"""

REAFINER_JUDGEMENT_USER_PROMPT = """
Question: {question}
Knowledge Graph (KG) context: {triples_string}
"""

REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT = """
As an advanced error abduction assistant, your task is to analyze the error reasons based on the given interaction history.

Analyze the reasons of the unanswerable questions based on the given interaction history. Output your analysis in the following format:
<abduction>...</abduction>

**Important:** You must think carefully about the interaction history before making your analysis. And output your analysis result directly in the specified format.
"""

REAFINER_ERROR_ABDUCTION_USER_PROMPT = """
Interaction history: {interaction_history}
"""

REAFINER_KG_REFINEMENT_SYSTEM_PROMPT = """
As an advanced knowledge graph refinement assistant, your task is to refine knowledge graph to make it more suitable for answering the given question.

Based on the given KG and the analysed error reasons, refine the given KG to replace the original KG, which makes it more easily for retrieval and answering the given question. Output the refined KG as a JSON array of triples in the following format:
<refinement>[{"subject": "...", "relation": "...", "object": "..."}, {"subject": "...", "relation": "...", "object": "..."}, ...]</refinement>

**Important:** You must think carefully about the given KG and the analysed error reasons before making your refinement. And output your refinement result directly in the specified format.
"""

REAFINER_KG_REFINEMENT_USER_PROMPT = """
KG: {triples_string}
Error reasons: {error_reasons}
"""

REFINE_SUBGRAPH_SYSTEM_PROMPT = """
As an advanced knowledge graph refinement assistant, your task is to refine knowledge graphs to make them more suitable for answering questions. Follow the three-step thinking process provided in the user instructions.

Refine the knowledge graph (KG) to make it more suitable for answering the given question. You must perform this task in three sequential thinking steps:

**Step 1: Answerable Judgement**
Evaluate whether the given question is answerable based on the provided KG context. If the answer is Yes, directly terminate the process. Output your judgment in the following format:
<judge>Yes</judge> or <judge>No</judge>

**Step 2: Error Abduction**
If the above answer is Yes, directly terminate the process. If the above answer is No, identify the specific problems in the KG. Analyze the KG for three types of errors: Redundant, Incompleteness, and Incorrectness. Output your analysis in the following format:
<abduction>...</abduction>

**Step 3: Refined KG Generation**
Based on your abduction analysis and the given KG, generate a refined KG that addresses the identified issues. Output the refined KG as a JSON array of triples in the following format:
<refinement>[{"subject": "...", "relation": "...", "object": "..."}, {"subject": "...", "relation": "...", "object": "..."}, ...]</refinement>

**Important:** You must follow this three-step thinking process and output each step in the specified format. Think carefully about each step before proceeding to the next.
"""

REFINE_SUBGRAPH_USER_PROMPT = """
Question: {question}
Knowledge Graph (KG) context: {triples_string}
"""