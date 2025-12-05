# Helix Navigator - Enhancing Biomedical Education with an LLM-Based Knowledge Graph Approach

**Learn LangGraph and Knowledge Graphs through Biomedical AI -- Enhancing Medical Education with LLM-Based Learning Tool**

An interactive educational project that teaches modern AI development through hands-on biomedical applications. Build AI agents that answer complex questions about genes, proteins, diseases, and drugs using graph databases and multi-step AI workflows. This educational tool was built over the existing Helix Navigator workflow.

*Navigate: [Getting Started](docs/getting-started.md) | [Foundations Guide](docs/foundations-and-background.md) | [Reference](docs/reference.md) | [Technical Guide](docs/technical-guide.md)*


## What You'll Learn

- **Knowledge Graphs**: Represent domain knowledge as nodes and relationships
- **LangGraph**: Build multi-step AI workflows with state management  
- **Cypher Queries**: Query graph databases effectively
- **AI Integration**: Combine language models with structured knowledge
- **Biomedical Applications**: Apply AI to drug discovery and personalized medicine

## Project Overview

**Research Question:** Can LLM-based knowledge graph systems improve medical students' understanding of biomedical relationships and evidence-based decision making?

**Key Added Features:**

**Learning & Visualization Features**

- 🎓 **Educational Mode (New)** – Toggle a dedicated *Learning Mode* that wraps every answer with pedagogical scaffolding: question difficulty level, step‑by‑step query decomposition, reasoning/justification, limitations, key vocabulary, and suggested follow‑up questions. Designed and implemented specifically to support medical students’ conceptual understanding and self‑directed learning.
- 🔬 **Comparative Analysis (New)** - Side-by-side entity comparisons ("Compare TP53 vs BRCA1")
- 📊 **Statistical Aggregation (New)** - Quantitative queries ("How many diseases per gene?")
- 💡 **Answer Justification (New)** - Transparent reasoning explanations
- 🎯 **6-Step LangGraph Workflow** - Classify → Extract → Generate → Execute → Score → Justify

**Dataset:** 500 genes, 661 proteins, 191 diseases, 350 drugs, 3,039 relationships

---

## Quick Start

1. **New to these concepts?** Read the [Foundations Guide](docs/foundations-and-background.md)
2. **Setup**: Follow [Getting Started](docs/getting-started.md) for installation
3. **Learn**: Use the interactive Streamlit web interface
4. **Practice**: Work through the exercises in the web app

## Technology Stack

- **LangGraph**: AI workflow orchestration
- **Neo4j**: Graph database
- **Anthropic Claude**: Language model
- **Streamlit**: Interactive web interface
- **LangGraph Studio**: Visual debugging

## Installation

**Quick Setup**: Python 3.10+, Neo4j, PDM

```bash
# Install dependencies
pdm install

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Load data and start
pdm run load-data
pdm run app
```

## Project Structure

```
├── LICENSE
├── README.md
├── pyproject.toml
├── pdm.lock
├── .env.example
├── .gitignore
│
├── data/
│   ├── genes.csv
│   ├── proteins.csv
│   ├── diseases.csv
│   ├── drugs.csv
│   └── relationships.csv
│   
│
├── docs/
│   ├── foundations-and-background.md
│   ├── reference.md
│   ├── getting-started.md
│   └──  technical_guide.md
|
│
├── scripts/
│   ├── load_data.py           # Build Neo4j graph from CSVs
│   └── quickstart.py
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── workflow_agent.py        # Main LangGraph workflow - edited to integrate educational mode etc.
│   │   ├── educational_agent.py     # Educational Mode: difficulty, decomposition, vocab,,
│   │   ├── graph_interface.py       #Simplified wrapper for Neo4j interactions with error handling
│   │   └── utils.py                 # Shared helpers for prompts, parsing, and state
│   │
│   │
│   ├── web/
│   │   ├── __init__.py
│   │   └── app.py                   # Streamlit UI, mode toggle, example questions, results & educational toggle
│   │
│   ├── .env             # personal environment/config management (API keys, Neo4j URI, etc.)
│   
│
├── tests/
│   ├── __init__.py
│   ├── test_graph_interface.py
│   ├── test_cypher_generation.py
│   └──  test_workflow_agent.py
```

**Key Files**:
- `src/agents/workflow_agent.py` - Main LangGraph agent
- `src/web/app.py` - Interactive Streamlit interface
- `src/agents/educational_agent.py` - Educational Mode agent
- `docs/` - Complete documentation

## Running the Application

Go to the top and follow the [Getting Started](docs/getting-started.md) for installation steps.

### Basic Usage
```bash
pdm run load-data         # Load biomedical data
pdm run app              # Start web interface
```

### Visual Debugging
```bash
pdm run langgraph    # Start LangGraph Studio
```

### Development
```bash
pdm run test            # Run tests (14 tests)
pdm run format          # Format code
pdm run lint            # Check quality
```
### Troubleshooting

### "Connection refused to Neo4j"
- Check if Neo4j is running  
  - Neo4j Desktop: Start your database  
  - Docker: `docker ps` (should see neo4j container)  
- Test connection

```bash
curl http://localhost:7474
```
### "Authentication failed"
- Verify password in `.env` matches Neo4j  
- Default: `NEO4J_PASSWORD=your_password`

### "No data found"
- Reload data  

```bash
pdm run load-data
```
- Verify data loaded  
- In Neo4j Browser:  
  ```
  MATCH (n) RETURN count(n)
  ```
- Should return > 1700 nodes

### "API key error"
- Check Anthropic API key in `.env`  
- Get key from: [https://console.anthropic.com/](https://console.anthropic.com/)

**Full commands**: See [Reference Guide](docs/reference.md)

## AI Agent

**WorkflowAgent** - LangGraph implementation with transparent processing for learning core LangGraph concepts through biomedical applications

## Example Questions

- **"Which drugs have high efficacy for treating diseases?"**
- **"Which approved drugs treat cardiovascular diseases?"**
- **"Which genes encode proteins that are biomarkers for diseases?"**
- **"What drugs target proteins with high confidence disease associations?"**
- **"Which approved drugs target specific proteins?"**
- **"Which genes are linked to multiple disease categories?"**
- **"What proteins have causal associations with diseases?"**
- **"How many diseases are in each category?"**
- **"Count the number of proteins each gene encodes"**
- **"What's the distribution of drug approval statuses?"**
- **"Compare TP53 and BRCA1 gene disease associations"**
- **"Compare cardiovascular vs oncological disease treatment coverage"**
- **"Which has more protein isoforms: TP73 or TP63?"**

