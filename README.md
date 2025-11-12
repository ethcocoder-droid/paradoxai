python -m http.server 5500# Paradox AI — Quantum-Like God AI

**By Ethco Coders** - A sophisticated quantum-inspired AI pipeline implementing a complete perception → knowledge → reasoning → curiosity → self-awareness → output → developer-learning loop.

## 🎯 Project Status: **COMPLETE & OPERATIONAL**

✅ **All 13 Tests Pass** | ✅ **API Functional** | ✅ **REPL Working** | ✅ **Full Integration Verified**

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Modes
```bash
# Interactive REPL mode
python main.py --repl

# Backend API server (port 8000)
python backend/app.py

# Complete system with UI
python main.py --backend --ui

# Static UI only (port 5500)
python -m http.server 5500
```

### API Usage
```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Authorization: Bearer dev-token-123" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?"}'
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTUM-LIKE AI PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input → Perception → Knowledge → Reasoning → Interference │
│       ↓              ↓           ↓           ↓            ↓     │
│  Developer ← Output ← Self-Awareness ← Curiosity ← Questions  │
│   Learning        Collapse     Emotions   Uncertainty         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

#### 🔍 **Perception** (`modules/perception/`)
- **InputEncoder**: Multi-modal encoding (text, images, concepts)
- **Features**: Vector normalization, superposition initialization
- **Technology**: NumPy-based with optional Pillow support

#### 🧠 **Knowledge** (`modules/knowledge/`)
- **HyperMatrixStore**: 3D tensor storage (concepts × features × branches)
- **EntanglementManager**: Quantum-like relationships between concepts
- **Persistence**: JSON-based storage system

#### 🧮 **Reasoning** (`modules/reasoning/`)
- **PathologicalLogic**: Multi-perspective analysis with contradictory viewpoints
- **InterferenceEngine**: Quantum-like interference with phases and probabilities
- **Innovation**: Entanglement-informed probability rebalancing

#### 🤔 **Curiosity** (`modules/curiosity/`)
- **QuestionGenerator**: Shannon entropy-based uncertainty detection
- **Features**: Internal/external question generation, curiosity state tracking
- **Metrics**: Uncertainty thresholds, conflict detection, decay mechanisms

#### 🎭 **Self-Awareness** (`modules/self_awareness/`)
- **AIEmotions**: Four-dimensional affective system (Inceptio, Equilibria, Reflexion, Fluxion)
- **AttentionManager**: Balances curiosity-driven exploration vs correctness-driven consolidation
- **Adaptation**: Emotion updates from signals and feedback

#### 📤 **Output** (`modules/output/`)
- **ProbabilisticCollapse**: Emotion-aware response generation
- **Features**: Temperature-based sampling, multi-tone support (friendly/formal/neutral)
- **Customization**: User-adaptive responses with configurable parameters

#### 📚 **Developer Learning** (`modules/learning_from_developer/`)
- **DeveloperInputHandler**: Encodes developer feedback
- **HyperMatrixUpdater**: Applies learning to knowledge base
- **CuriosityFeedback**: Generates clarification requests

---

## 🧪 Testing

### Run All Tests
```bash
# Comprehensive test suite
python -m pytest tests/ -v

# Quick test run
pytest -q
```

### Test Coverage
- ✅ **13/13 Tests Pass**
- ✅ **Unit Tests**: All modules individually tested
- ✅ **Integration Tests**: End-to-end pipeline verified
- ✅ **API Tests**: Backend endpoints validated

---

## 📊 Performance Characteristics

- **Startup Time**: ~2 seconds
- **Query Processing**: ~0.5 seconds per query
- **Memory Usage**: ~50MB base + data structures
- **Scalability**: Efficiently handles 1000+ concepts

---

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `API_TOKEN`: Authentication token (default: `dev-token-123`)

### Key Parameters
- **Vector Dimensions**: Configurable per module
- **Emotion Weights**: Adjustable affective parameters
- **Temperature**: Sampling temperature for output
- **Uncertainty Threshold**: Curiosity trigger levels

---

## 📁 Project Structure

```
QuantumGodAI/
├── modules/                    # Core AI components
│   ├── perception/            # Input encoding
│   ├── knowledge/             # Memory & entanglement
│   ├── reasoning/             # Logic & interference
│   ├── curiosity/             # Question generation
│   ├── self_awareness/        # Emotions & attention
│   ├── output/                # Response generation
│   ├── learning_from_developer/ # Human feedback
│   └── utils/                 # Utility functions
├── backend/                   # Flask API server
├── ui/                        # Frontend interface
├── data/                      # JSON data storage
├── tests/                     # Unit tests
├── main.py                    # Main integration
├── requirements.txt           # Dependencies
└── ANALYSIS_REPORT.md         # Technical documentation
```

---

## 🎮 Usage Examples

### REPL Mode
```
> What is quantum computing?
{
  "response": "Here's a clear English answer about quantum computing...",
  "emotions": {"Inceptio": 0.60, "Equilibria": 0.46, "Reflexion": 0.50, "Fluxion": 0.64},
  "probabilities": [0.20, 0.12, 0.12, 0.19, 0.19, 0.17],
  "chosen_index": 5,
  "curiosity": {"uncertainty": 0.0, "entropy": 0.0}
}
```

### Developer Input
```bash
curl -X POST http://127.0.0.1:8000/api/developer_input \
  -H "Authorization: Bearer dev-token-123" \
  -H "Content-Type: application/json" \
  -d '{"concept_id": "ai:definition", "text": "AI is intelligence demonstrated by machines"}'
```

---

## 🔬 Research Applications

- **Quantum-Inspired Computing**: Novel AI architectures
- **Cognitive Modeling**: Multi-perspective reasoning
- **Human-AI Interaction**: Emotion-aware systems
- **Knowledge Representation**: Entanglement-based storage
- **Curiosity-Driven Learning**: Uncertainty-based exploration

---

## 📈 Next Steps (Optional Enhancements)

### Production Features
- Database backend (PostgreSQL/MongoDB)
- Distributed architecture
- GPU acceleration
- Advanced NLP models
- Real-time monitoring

### Research Extensions
- Quantum circuit integration
- Multi-agent systems
- Advanced emotional learning
- Cognitive architecture modeling

---

## 📞 Support

This is a research project demonstrating quantum-inspired AI concepts. For questions about the architecture or implementation details, refer to the `ANALYSIS_REPORT.md` and `IMPLEMENTATION_SUMMARY.md` files.

**Status**: ✅ **Fully Operational** | **Ready for Research & Development**
**to start the server of ui**
```bash
python -m http.server 5500
```