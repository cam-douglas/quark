# 📚 Comprehensive Validation System Reference Card

## ✅ SYSTEM OVERVIEW

The Comprehensive Validation System **MANDATORILY** uses **ALL 20+ API sources** from `/Users/camdouglas/quark/data/credentials/all_api_keys.json` plus **79+ open access sources** to validate claims and reduce overconfidence.

## 🎯 INTELLIGENT SOURCE SELECTION

The system automatically categorizes queries and selects the BEST validation sources:

### Query Categories → Source Mapping

| Query Type | Primary Sources | Confidence Weight |
|------------|----------------|-------------------|
| **Protein Structure** | RCSB PDB (0.98), UniProt (0.96), AlphaFold (0.95) | Very High |
| **Genomics** | NCBI E-utilities (0.98), Ensembl (0.95), BLAST (0.94) | Very High |
| **Chemistry** | PubChem (0.95) | High |
| **Materials Science** | Materials Project (0.92), OQMD (0.90) | High |
| **Neuroscience** | PubMed Central (0.95), NCBI E-utilities (0.98) | Very High |
| **Machine Learning** | OpenML (0.88), Hugging Face (0.85), Kaggle (0.82) | High |
| **Scientific Literature** | PubMed (0.95), ArXiv (0.90) | High |
| **Clinical/Medical** | PubMed Central (0.95), NCBI E-utilities (0.98) | Very High |
| **Computational/Math** | Wolfram Alpha (0.92), ArXiv (0.90) | High |
| **Code Documentation** | GitHub (0.88), Context7 (0.85) | High |
| **General AI Validation** | Claude (0.82), OpenAI (0.80), Gemini (0.78) | Medium-High |

## 🔐 ALL AVAILABLE VALIDATION SOURCES

### Tier 1: Authoritative Scientific Databases (Confidence: 0.94-0.98)
- **RCSB PDB** - Experimental protein structures
- **NCBI E-utilities** - PubMed, GenBank, Gene, Protein databases  
- **UniProt** - Protein sequences and functional annotation
- **PubMed Central** - Biomedical literature
- **Ensembl** - Genomics data across vertebrates
- **PubChem** - Chemical compounds and bioassays

### Tier 2: Specialized Scientific Resources (Confidence: 0.90-0.95)
- **AlphaFold** - AI-predicted protein structures (200M+ structures)
- **BLAST** - Sequence similarity searches
- **Materials Project** - Computed materials properties (API: O0oXYcZo6YumgKUOzDJCx9mFiAk9pP4l)
- **Wolfram Alpha** - Computational knowledge engine
- **ArXiv** - Preprint archive (2M+ papers)
- **OQMD** - Quantum materials database (700K+ materials)

### Tier 3: ML/Data Science Resources (Confidence: 0.82-0.88)
- **OpenML** - ML datasets and experiments (20K+ datasets)
- **GitHub** - Code repositories and documentation  
- **Context7** - Library documentation for AI
- **Hugging Face** - ML models and datasets
- **Kaggle** - Datasets and competitions

### Tier 4: AI Validation (Confidence: 0.78-0.82)
- **Claude** - Advanced reasoning validation
- **OpenAI GPT** - General AI validation
- **Gemini** - Google AI validation
- **OpenRouter** - Multi-LLM validation

## 📋 MANDATORY USAGE

### Quick Implementation
```python
from tools_utilities.comprehensive_validation_system import mandatory_validate

# ALWAYS validate before making claims
result = mandatory_validate(
    claim="Your technical statement here",
    context="Additional context if needed"
)

# Check validation results
confidence = result['confidence']  # 0.0 to 0.9 (capped)
consensus = result['consensus']    # STRONG_SUPPORT, MODERATE_SUPPORT, MIXED_EVIDENCE, CONTRADICTED
sources_checked = result['sources_checked']  # Number of sources validated against

# Express appropriate confidence
if confidence < 0.4:
    prefix = "⚠️ LOW CONFIDENCE"
elif confidence < 0.7:
    prefix = "🟡 MEDIUM CONFIDENCE"  
else:
    prefix = "✅ HIGH CONFIDENCE (capped at 90%)"
```

### System Initialization
```python
from tools_utilities.comprehensive_validation_system import get_validation_system

# Get singleton instance (loads ALL sources automatically)
system = get_validation_system()

# Check what sources are available
all_sources = system.get_all_available_sources()
# Returns: ['alphafold', 'rcsb_pdb', 'ensembl', 'ncbi_eutils', 'pubchem', ...]

# Verify all sources are active
status = system.verify_all_sources_active()
# Returns: {'alphafold': True, 'rcsb_pdb': True, ...}
```

## 🎯 INTELLIGENT CATEGORY DETECTION

The system automatically detects query categories based on keywords:

### Keyword → Category Mapping
- **Protein/Structure**: protein, structure, pdb, fold, domain, binding site
- **Genomics**: gene, genome, transcript, chromosome, variant, snp, mutation
- **Neuroscience**: brain, neuron, synapse, neural, cognitive, cortex
- **Chemistry**: compound, molecule, drug, chemical, reaction, inhibitor
- **Materials**: material, crystal, band gap, lattice, alloy, semiconductor
- **ML/AI**: machine learning, ml, model, dataset, training, neural network
- **Literature**: paper, publication, study, research, citation, reference
- **Sequences**: sequence, blast, alignment, homolog, ortholog, fasta
- **Clinical**: disease, patient, clinical, therapy, treatment, diagnosis
- **Computational**: calculate, compute, equation, formula, algorithm
- **Code**: code, api, function, library, documentation, implement
- **Physics**: physics, quantum, particle, energy, force, field
- **Math**: math, equation, theorem, proof, calculus, algebra

## 📊 VALIDATION CONSENSUS LEVELS

| Consensus Level | Meaning | Action Required |
|-----------------|---------|-----------------|
| **STRONG_SUPPORT** | All sources agree | Can state with high confidence (cap at 90%) |
| **MODERATE_SUPPORT** | >70% sources agree | Express moderate confidence, note minority views |
| **MIXED_EVIDENCE** | 30-70% agreement | Must express significant uncertainty |
| **CONTRADICTED** | <30% agreement | Should not make claim, seek clarification |
| **NO_VALIDATION** | No sources available | Cannot proceed without validation |

## 🚨 ENFORCEMENT RULES

1. **NEVER** make claims without running `mandatory_validate()`
2. **ALWAYS** check at least 3 independent sources
3. **NEVER** express >90% confidence even with perfect validation
4. **ALWAYS** disclose which sources were checked
5. **ALWAYS** express uncertainty when consensus is not strong

## 📁 File Locations

- **Validation System**: `/Users/camdouglas/quark/tools_utilities/comprehensive_validation_system.py`
- **Credentials**: `/Users/camdouglas/quark/data/credentials/all_api_keys.json`
- **Knowledge Sources**: `/Users/camdouglas/quark/data/knowledge/validation_system/open_access_literature_sources.json`
- **Anti-Overconfidence Rule**: `/Users/camdouglas/quark/.cursor/rules/anti-overconfidence-validation.mdc`

## 🔄 System Features

- **Intelligent Source Selection**: Automatically picks best sources for query type
- **Caching**: Prevents redundant validations
- **History Tracking**: Logs all validations for audit
- **Category Detection**: 15 query categories supported
- **Cross-Validation**: Always uses multiple sources
- **Confidence Weighting**: Sources have different reliability weights
- **Consensus Building**: Aggregates multiple source opinions

## 📈 Statistics

- **Total API Sources**: 20+ configured and active
- **Open Access Sources**: 79+ literature databases
- **Query Categories**: 15 intelligent categories
- **Confidence Range**: 0-90% (hard capped)
- **Minimum Sources**: 3 required for any validation
- **Source Reliability**: 0.78-0.98 confidence weights

---

**REMEMBER**: This system is MANDATORY for all technical claims. The anti-overconfidence rule at priority 0 requires using this comprehensive validation system that checks ALL available resources intelligently.
