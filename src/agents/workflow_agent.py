"""
LangGraph workflow agent for biomedical knowledge graphs.

5-step workflow: Classify → Extract → Generate → Execute → Format
"""

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .graph_interface import GraphInterface


class WorkflowState(TypedDict):
    """State that flows through the workflow steps."""

    user_question: str
    question_type: Optional[str]
    entities: Optional[List[str]]
    cypher_query: Optional[str]
    results: Optional[List[Dict]]
    final_answer: Optional[str]
    error: Optional[str]
    justification: Optional[str]


class WorkflowAgent:
    """LangGraph workflow agent for biomedical knowledge graphs."""

    # Class constants
    MODEL_NAME = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 200

    # Default schema query
    SCHEMA_QUERY = (
        "MATCH (n) RETURN labels(n) as node_type, count(n) as count "
        "ORDER BY count DESC LIMIT 10"
    )

    def __init__(self, graph_interface: GraphInterface, anthropic_api_key: str):
        self.graph_db = graph_interface
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.schema = self.graph_db.get_schema_info()
        self.property_values = self._get_key_property_values()
        self.workflow = self._create_workflow()

    def _get_key_property_values(self) -> Dict[str, List[Any]]:
        """Get property values dynamically from all nodes and relationships.

        This method discovers all available properties in the database schema and
        collects sample values for each property. This enables the LLM to generate
        more accurate queries by knowing what property values actually exist.

        Returns:
            Dict mapping property names to lists of sample values from the database
        """
        values = {}
        try:
            # Discover and collect property values from all node types in the database
            # This replaces hardcoded property lists with dynamic schema discovery
            for node_label in self.schema.get("node_labels", []):
                # Get all properties that exist for this node type
                node_props = self.schema.get("node_properties", {}).get(node_label, [])
                for prop_name in node_props:
                    # Avoid duplicate property names (same property might exist
                    # on multiple node types)
                    if prop_name not in values:
                        # Query the database for actual property values
                        # (limited to 20 for performance)
                        prop_values = self.graph_db.get_property_values(
                            node_label, prop_name
                        )
                        # Only store properties that have actual values in the database
                        if prop_values:
                            values[prop_name] = prop_values

            # Discover and collect property values from all relationship types
            # This ensures we capture relationship-specific properties like
            # confidence, weight, etc.
            for rel_type in self.schema.get("relationship_types", []):
                # GraphInterface expects 'REL_' prefix for relationship queries
                rel_label = f"REL_{rel_type}"
                # Get all properties that exist for this relationship type
                rel_props = self.schema.get("relationship_properties", {}).get(
                    rel_type, []
                )
                for prop_name in rel_props:
                    # Skip if we already have this property from a node type
                    if prop_name not in values:
                        try:
                            # Query relationship properties using the REL_ prefix
                            # convention
                            prop_values = self.graph_db.get_property_values(
                                rel_label, prop_name
                            )
                            # Only store if the relationship actually has values
                            # for this property
                            if prop_values:
                                values[prop_name] = prop_values
                        except Exception:
                            # Some relationships might not have certain properties -
                            # skip gracefully
                            continue

        except Exception:
            pass
        return values

    def _get_llm_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Get response from LLM and handle content extraction."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        try:
            response = self.anthropic.messages.create(
                model=self.MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0]
            return content.text.strip() if hasattr(content, "text") else str(content)
        except Exception as e:
            raise RuntimeError(f"LLM response failed: {str(e)}")

    def _create_workflow(self) -> Any:
        """Create the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("classify", self.classify_question)
        workflow.add_node("extract", self.extract_entities)
        workflow.add_node("generate", self.generate_query)
        workflow.add_node("execute", self.execute_query)
        workflow.add_node("format", self.format_answer)
        workflow.add_node("justify", self.add_justification)

        workflow.add_edge("classify", "extract")
        workflow.add_edge("extract", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "format")
        workflow.add_edge("format", "justify")  # NEW edge for justify
        workflow.add_edge("justify", END)

        workflow.set_entry_point("classify")
        return workflow.compile()

    def _build_classification_prompt(self, question: str) -> str:
        """Build classification prompt with consistent formatting."""
        return f"""Classify this biomedical question. Choose one:
- gene_disease: genes and diseases
- drug_treatment: drugs and treatments
- protein_function: proteins and functions
- statistical_analysis: counting, aggregating, or analyzing distributions
- comparative_analysis: comparing two or more entities side-by-side
- general_db: database exploration
- general_knowledge: biomedical concepts

Question: {question}

Respond with just the type."""

    def classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the biomedical question type using an LLM.

        Uses LLM-based classification instead of hardcoded keyword matching for
        more flexible and accurate question type detection. This enables the agent
        to handle nuanced questions that don't fit simple keyword patterns.
        """
        try:
            # Build classification prompt with available question types
            prompt = self._build_classification_prompt(state["user_question"])
            # Use minimal tokens since we only need a single classification word
            state["question_type"] = self._get_llm_response(prompt, max_tokens=20)
        except Exception as e:
            # If classification fails, record error but continue with safe fallback
            state["error"] = f"Classification failed: {str(e)}"
            # Default to general knowledge to avoid database queries with
            # malformed inputs
            state["question_type"] = "general_knowledge"
        return state

    def extract_entities(self, state: WorkflowState) -> WorkflowState:
        """Extract biomedical entities from the question.

        Uses the database schema to guide entity extraction, ensuring that
        only entities that can actually be found in the database are
        extracted.
        This improves query generation accuracy by providing relevant context.
        """
        # Skip entity extraction for questions that don't need
        # database-specific entities
        question_type = state.get("question_type")
        if question_type in ["general_db", "general_knowledge", "statistical_analysis"]:
            # General questions don't need specific entity extraction
            state["entities"] = []
            return state

        # Build dynamic property information from actual database content
        # This replaces hardcoded examples with real data from the database
        property_info = []
        for prop_name, values in self.property_values.items():
            if values:  # Only show properties with actual values in database
                # Show first 3 values as representative examples for the LLM
                sample_values = ", ".join(str(v) for v in values[:3])
                property_info.append(f"- {prop_name}: {sample_values}")

        entity_types_str = ", ".join(self.schema.get("node_labels", []))
        relationship_types_str = ", ".join(self.schema.get("relationship_types", []))

        prompt = (
            f"""Extract biomedical terms and concepts from this question """
            f"""based on the database schema:

Available entity types: {entity_types_str}
Available relationships: {relationship_types_str}

Available property values in database:
{chr(10).join(property_info) if property_info else "- No property values available"}

Question: {state['user_question']}

Extract ALL relevant terms including:
- Specific entity names mentioned
- Entity types referenced
- Property values or constraints
- Relationship concepts
- General biomedical concepts

Return a JSON list: ["term1", "term2"] or []"""
        )

        try:
            response_text = self._get_llm_response(prompt, max_tokens=100)

            # Clean up response text before JSON parsing
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = (
                    cleaned_text.replace("```json", "").replace("```", "").strip()
                )

            state["entities"] = json.loads(cleaned_text)
        except (json.JSONDecodeError, AttributeError):
            # Fallback to empty list if JSON parsing fails
            state["entities"] = []

        return state

    def generate_query(self, state: WorkflowState) -> WorkflowState:
        """Generate Cypher query based on question type.

        Creates database queries dynamically using the actual schema and property
        values discovered from the database. This ensures queries are valid and
        use only properties/relationships that actually exist.
        """
        question_type = state.get("question_type", "general")

        # Database exploration questions use a simple schema overview query
        if question_type == "general_db":
            # Use predefined query to show database structure and content overview
            state["cypher_query"] = self.SCHEMA_QUERY
            return state

        # General knowledge questions don't need database queries
        if question_type == "general_knowledge":
            # Skip database query for conceptual questions that don't need data lookup
            state["cypher_query"] = None
            return state
        # Statistical analysis questions need aggregation queries
        if question_type == "statistical_analysis":
            # Build aggregation-focused prompt with COUNT, AVG, SUM capabilities
            stat_prompt = f"""Create a Cypher aggregation query for this analytical question:

        Question: {state['user_question']}

        Database Schema:
        Nodes: {', '.join(self.schema['node_labels'])}
        Relationships: {', '.join(self.schema['relationship_types'])}

        Available properties to group by or aggregate:
        {json.dumps(self.schema['node_properties'], indent=2)}

        Available relationship properties:
        {json.dumps(self.schema['relationship_properties'], indent=2)}

        AGGREGATION PATTERNS:
        1. COUNT groups: MATCH (n:Label) RETURN n.property, count(*) as count ORDER BY count DESC
        2. Distribution: MATCH (n)-[r:REL]->(m) RETURN type(r), count(*) as count
        3. Category stats: MATCH (n:Label) WHERE n.category IS NOT NULL RETURN n.category, count(*) as total
        4. Average/Sum: MATCH (n:Node) RETURN avg(n.numeric_property), sum(n.numeric_property)
        5. Top N: Use ORDER BY count DESC LIMIT 10

        Return ONLY the Cypher query. Focus on COUNT, GROUP BY (via RETURN), and ORDER BY."""

            cypher_query = self._get_llm_response(stat_prompt, max_tokens=200)

            # Clean up markdown formatting
            if cypher_query.startswith("```"):
                cypher_query = "\n".join(
                    line
                    for line in cypher_query.split("\n")
                    if not line.startswith("```") and not line.startswith("cypher")
                ).strip()

            state["cypher_query"] = cypher_query
            return state
        # Comparative analysis questions need side-by-side comparison queries
        if question_type == "comparative_analysis":
            entities = state.get("entities", [])

            # Build comparison-focused prompt
            # Build relationship guide
            relationship_guide = """
        RELATIONSHIP SCHEMA (IMPORTANT - Use correct relationships):
        - Gene -[:ENCODES]-> Protein
        - Gene -[:LINKED_TO]-> Disease  
        - Protein -[:ASSOCIATED_WITH]-> Disease
        - Drug -[:TREATS]-> Disease
        - Drug -[:TARGETS]-> Protein

        PROPERTY NAMES (Use these exact names):
        - Genes: gene_name, chromosome, function, expression_level
        - Proteins: protein_name, gene_id, molecular_weight, structure_type
        - Diseases: disease_name, category, prevalence, severity
        - Drugs: drug_name, type, approval_status, mechanism
        """

            comparison_prompt = f"""Create a Cypher query that compares entities side-by-side:

        Question: {state['user_question']}
        Entities to compare: {entities}

        {relationship_guide}

        COMPARISON QUERY PATTERNS:

        1. Compare two GENES by disease associations:
        MATCH (g1:Gene {{gene_name: 'TP53'}})-[:LINKED_TO]->(d1:Disease)
        WITH count(DISTINCT d1) as count1, 'TP53' as entity1
        MATCH (g2:Gene {{gene_name: 'BRCA1'}})-[:LINKED_TO]->(d2:Disease)
        WITH count(DISTINCT d2) as count2, 'BRCA1' as entity2, count1, entity1
        RETURN entity1, count1, entity2, count2

        2. Compare two DRUGS by protein targets:
        MATCH (dr1:Drug {{drug_name: 'Lisinopril'}})-[:TARGETS]->(p1:Protein)
        WITH count(DISTINCT p1) as count1, 'Lisinopril' as entity1
        MATCH (dr2:Drug {{drug_name: 'Metoprolol'}})-[:TARGETS]->(p2:Protein)
        WITH count(DISTINCT p2) as count2, 'Metoprolol' as entity2, count1, entity1
        RETURN entity1, count1, entity2, count2

        3. Compare DISEASE CATEGORIES by treatment count:
        MATCH (d1:Disease {{category: 'cardiovascular'}})<-[:TREATS]-(dr1:Drug)
        WITH count(DISTINCT dr1) as drugs1, count(DISTINCT d1) as diseases1, 'cardiovascular' as cat1
        MATCH (d2:Disease {{category: 'oncological'}})<-[:TREATS]-(dr2:Drug)
        WITH count(DISTINCT dr2) as drugs2, count(DISTINCT d2) as diseases2, 'oncological' as cat2, drugs1, diseases1, cat1
        RETURN cat1, drugs1, diseases1, cat2, drugs2, diseases2

        4. Compare two PROTEINS by disease associations:
        MATCH (p1:Protein {{protein_name: 'TP53'}})-[:ASSOCIATED_WITH]->(d1:Disease)
        WITH count(DISTINCT d1) as count1, 'TP53' as entity1
        MATCH (p2:Protein {{protein_name: 'BRCA1'}})-[:ASSOCIATED_WITH]->(d2:Disease)
        WITH count(DISTINCT d2) as count2, 'BRCA1' as entity2, count1, entity1
        RETURN entity1, count1, entity2, count2

        CRITICAL RULES:
        - Gene-Disease: Use LINKED_TO (NOT ASSOCIATED_WITH)
        - Protein-Disease: Use ASSOCIATED_WITH
        - Use exact property names: gene_name, disease_name, drug_name, protein_name
        - Always use DISTINCT when counting
        - Return entity names and counts for comparison

        Return ONLY the Cypher query."""

            cypher_query = self._get_llm_response(comparison_prompt, max_tokens=300)

            # Clean up markdown formatting
            if cypher_query.startswith("```"):
                cypher_query = "\n".join(
                    line
                    for line in cypher_query.split("\n")
                    if not line.startswith("```") and not line.startswith("cypher")
                ).strip()

            state["cypher_query"] = cypher_query
            return state

        # Build dynamic relationship guide from actual database schema
        # This replaces hardcoded relationship patterns with discovered relationships
        relationship_guide = f"""
Available relationships:
{' | '.join([f'- {rel}' for rel in self.schema['relationship_types']])}"""

        # Build comprehensive property information from database discovery
        # This gives the LLM concrete examples of what property values exist
        property_details = []
        for prop_name, values in self.property_values.items():
            if values:  # Only include properties with actual values in the database
                # Auto-detect value type to help LLM understand data format
                value_type = (
                    "text values" if isinstance(values[0], str) else "numeric values"
                )
                property_details.append(f"- {prop_name}: {values} ({value_type})")

        property_info = f"""Property names and values:
Node properties: {self.schema['node_properties']}
Available property values:
{chr(10).join(property_details) if property_details else "- No values available"}
Use WHERE property IN [value1, value2] for filtering."""
        prompt = f"""Create a Cypher query for this biomedical question:

Question: {state['user_question']}
Type: {question_type}
Schema:
Nodes: {', '.join(self.schema['node_labels'])}
Relations: {', '.join(self.schema['relationship_types'])}
{property_info}
{relationship_guide}
Entities: {state.get('entities', [])}

Use MATCH, WHERE with CONTAINS for filtering, RETURN, LIMIT 10.
IMPORTANT: Use property names from schema above and IN filtering for property values.
Return only the Cypher query."""

        cypher_query = self._get_llm_response(prompt, max_tokens=150)

        # Clean up LLM response formatting (remove markdown code blocks)
        # LLMs often wrap code in ```cypher blocks, so we need to extract just the query
        if cypher_query.startswith("```"):
            cypher_query = "\n".join(
                line
                for line in cypher_query.split("\n")
                # Remove markdown code block markers and language specifiers
                if not line.startswith("```") and not line.startswith("cypher")
            ).strip()

        state["cypher_query"] = cypher_query
        return state

    def execute_query(self, state: WorkflowState) -> WorkflowState:
        """Execute the generated Cypher query against the database.

        Safely executes the LLM-generated query with error handling to prevent
        crashes from malformed queries while capturing useful error information.
        """
        try:
            query = state.get("cypher_query")
            # Execute query only if one was generated (some question types
            # skip this step)
            state["results"] = self.graph_db.execute_query(query) if query else []
        except Exception as e:
            # Capture query execution errors but continue workflow to provide
            # helpful feedback
            state["error"] = f"Query failed: {str(e)}"
            # Set empty results so the format step can handle the error gracefully
            state["results"] = []

        return state

    def format_answer(self, state: WorkflowState) -> WorkflowState:
        """Format database results into human-readable answer.

        Takes raw database results and converts them into natural language
        responses, handling different question types and error conditions.
        """
        # Handle any errors that occurred during the workflow
        if state.get("error"):
            state["final_answer"] = (
                f"Sorry, I had trouble with that question: {state['error']}"
            )
            return state

        question_type = state.get("question_type")

        # General knowledge questions use LLM knowledge instead of database results
        if question_type == "general_knowledge":
            # Generate answer from LLM's training knowledge rather than database lookup
            state["final_answer"] = self._get_llm_response(
                f"""Answer this general biomedical question using your knowledge:

Question: {state['user_question']}

Provide a clear, informative answer about biomedical concepts.""",
                max_tokens=300,  # Allow more tokens for explanatory content
            )
            return state
        # Statistical analysis questions need formatted aggregation results
        if question_type == "statistical_analysis":
            results = state.get("results", [])
            if not results:
                state["final_answer"] = "No statistical data found for that query."
                return state

            # Format aggregation results as a clear summary
            state["final_answer"] = self._get_llm_response(
                f"""Format these statistical results into a clear, readable summary:

        Question: {state['user_question']}
        Results: {json.dumps(results[:10], indent=2)}
        Total rows: {len(results)}

        Create a concise summary highlighting:
        - Key statistics (totals, averages, top results)
        - Notable patterns or distributions
        - Top 5-10 results if applicable

        Use bullet points and be specific with numbers.""",
                max_tokens=300,
            )
            return state
        # Comparative analysis questions need side-by-side formatting
        if question_type == "comparative_analysis":
            results = state.get("results", [])
            if not results:
                state["final_answer"] = "No comparison data found for that query."
                return state

            # Format comparison results with clear contrast
            state["final_answer"] = self._get_llm_response(
                f"""Format these comparison results into a clear side-by-side summary:

    Question: {state['user_question']}
    Results: {json.dumps(results, indent=2)}

    Create a comparison summary that:
    1. Clearly identifies what is being compared
    2. Shows metrics side-by-side (use "vs" or "compared to")
    3. Highlights key differences or similarities
    4. Uses bullet points or tables for clarity

    Example format:
    "Here's the comparison of X vs Y:

    **X:**
    • Metric 1: value
    • Metric 2: value

    **Y:**
    • Metric 1: value (+/- difference from X)
    • Metric 2: value (+/- difference from X)

    **Key findings:** X has more of A, but Y has higher B..."

    Be specific and use actual numbers from the results.""",
                max_tokens=350,
            )
            return state

        # Handle database-based answers using query results
        results = state.get("results", [])
        if not results:
            # No results found - provide helpful guidance for next steps
            state["final_answer"] = (
                "I didn't find any information for that question. Try asking about "
                "genes, diseases, or drugs in our database."
            )
            return state

        # Convert raw database results into natural language using LLM
        state["final_answer"] = self._get_llm_response(
            f"""Convert these database results into a clear answer:

Question: {state['user_question']}
Results: {json.dumps(results[:5], indent=2)}
Total found: {len(results)}

Make it concise and informative.""",
            max_tokens=250,  # Balanced token limit for informative but concise
            # responses
        )
        return state

    def add_justification(self, state: WorkflowState) -> WorkflowState:
        """Add reasoning explanation to help users understand the workflow.

        Generates a 2-3 sentence explanation of:
        1. How the question was interpreted
        2. What database path was traversed
        3. Why the results answer the question

        This increases transparency and helps students learn reasoning process.
        """
        # Skip justification if there was an error
        if state.get("error"):
            return state

        # Build justification prompt
        justification_prompt = f"""Explain the reasoning behind this answer in 2-3 clear, concise sentences:

Question: {state['user_question']}
Question Type: {state['question_type']}
Extracted Entities: {state.get('entities', [])}
Query Generated: {state.get('cypher_query', 'N/A')}
Results Found: {len(state.get('results', []))} results

Explain:
1. How you interpreted the question type
2. What database relationships you searched (e.g., Gene→Protein→Disease)
3. Why these results answer the question

Format: "I interpreted this as a [type] question about [topic]. I searched the database by traversing [relationships] and filtering for [constraints]. These results show [key finding] which answers the question."

Keep it concise and educational - help the user understand your reasoning process."""

        try:
            justification = self._get_llm_response(justification_prompt, max_tokens=200)
            state["justification"] = justification
        except Exception as e:
            # If justification fails, don't block the workflow
            state["justification"] = f"Unable to generate explanation: {str(e)}"

        return state

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a biomedical question using the LangGraph workflow."""

        initial_state = WorkflowState(
            user_question=question,
            question_type=None,
            entities=None,
            cypher_query=None,
            results=None,
            final_answer=None,
            error=None,
            justification=None,
        )

        final_state = self.workflow.invoke(initial_state)

        return {
            "answer": final_state.get("final_answer", "No answer generated"),
            "question_type": final_state.get("question_type"),
            "entities": final_state.get("entities", []),
            "cypher_query": final_state.get("cypher_query"),
            "results_count": len(final_state.get("results", [])),
            "raw_results": final_state.get("results", [])[:3],
            "error": final_state.get("error"),
            "justification": final_state.get("justification"),
        }


def create_workflow_graph() -> Any:
    """Factory function for LangGraph Studio."""
    load_dotenv()

    graph_interface = GraphInterface(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    agent = WorkflowAgent(
        graph_interface=graph_interface,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    return agent.workflow
