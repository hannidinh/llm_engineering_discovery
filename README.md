# LLM Engineering Discovery

This repository contains my personal learning journey and explorations based on **Ed Donner's "Mastering LLM Engineering"** course.

🎓 **Course Repository**: https://github.com/ed-donner/llm_engineering

## 📖 About This Repository

This is a personal discovery and learning repository where I explore concepts, complete assignments, and experiment with Large Language Model engineering techniques from the comprehensive 8-week course by Ed Donner.

### Course Overview

The original course covers:

- **Week 1**: Foundations & Setup
- **Week 2**: Core LLM Concepts
- **Week 3**: Advanced Techniques
- **Week 4**: Vector Databases & RAG
- **Week 5**: LLM Agents & Function Calling
- **Week 6**: Multi-Agent Systems
- **Week 7**: Production & Deployment
- **Week 8**: Advanced Topics & Capstone

## 📁 Repository Structure

- `assignments/` – Final versions of assignment notebooks
- `notebooks/` – Exploratory work and research notes
- `prompts/` – Saved prompt templates and variations
- `data/` – Input/output data files and datasets
- `experiments/` – Personal experiments and proof-of-concepts

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- Git
- API keys for various LLM providers (OpenAI, Anthropic, etc.)

### 1. Clone and Setup Environment

```bash
# Clone this repository
git clone <your-repo-url>
cd llm_engineering_discovery

# Create Python environment using Anaconda
conda create -n llm-engineering python=3.11
conda activate llm-engineering

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the `.env.example` file to `.env` and configure your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required environment variables:

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOOGLE_API_KEY` - Google AI API key
- Other provider keys as needed

### 3. Start Jupyter Lab

```bash
jupyter lab
```

## 🔧 Tools & Technologies

This repository utilizes the following key technologies from the course:

- **LLM Providers**: OpenAI, Anthropic, Google AI, Ollama
- **Frameworks**: LangChain, LlamaIndex, Transformers
- **Vector Databases**: ChromaDB, Pinecone
- **Development**: Jupyter Lab, Python, Git
- **Deployment**: Modal, Gradio, Streamlit

## 📚 Learning Resources

- **Original Course**: [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering)
- **Setup Guides**: Available for Mac, PC, and Linux in the original repo
- **Community**: Course Discord and community contributions
- **Slides & Resources**: Course materials and presentations

## 🎯 Learning Objectives

Through this repository, I aim to:

- Master fundamental LLM engineering concepts
- Build practical applications using LLMs
- Understand RAG (Retrieval-Augmented Generation) systems
- Develop multi-agent systems
- Learn production deployment strategies
- Explore cutting-edge LLM techniques

## 🙏 Acknowledgments

Special thanks to **Ed Donner** for creating this comprehensive LLM Engineering course. This repository is built upon the excellent foundation and curriculum provided in his course.

## 📄 License

This repository is for educational purposes. Please refer to the original course repository for licensing information regarding course materials.

---

_This is a personal learning repository based on Ed Donner's LLM Engineering course. For the official course materials, please visit the [original repository](https://github.com/ed-donner/llm_engineering)._
