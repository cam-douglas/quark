# 📚 Wikipedia Training Guide for Brain Simulation

## Overview

This guide provides comprehensive options for training your model on Wikipedia data, addressing the limitations of the MediaWiki API and presenting better alternatives for full Wikipedia database training.

## ❌ MediaWiki API Limitations

The [MediaWiki API](https://www.mediawiki.org/wiki/API:Properties#API_documentation) you referenced has significant limitations for training:

### What the API Provides:
- ✅ Page content, categories, links, and metadata
- ✅ Real-time access to current Wikipedia content
- ✅ JSON format responses
- ✅ Multiple language support

### Critical Limitations for Training:
- ❌ **Rate Limits**: ~500 requests/second for bots
- ❌ **No Bulk Downloads**: Would require millions of API calls
- ❌ **Incomplete Data**: Missing images, templates, some metadata
- ❌ **Time Intensive**: Downloading full Wikipedia would take months
- ❌ **Resource Heavy**: Requires massive bandwidth and storage

### Example API Call:
```python
import requests

# Get a single page
url = "https://en.wikipedia.org/api/rest_v1/page/summary/Artificial_intelligence"
response = requests.get(url)
data = response.json()
print(data['extract'])  # Just the summary, not full content
```

## ✅ Better Options for Wikipedia Training

### 1. **Official Wikipedia Dumps** (Recommended for Complete Training)

**What**: Complete Wikipedia database dumps in XML format
**Size**: ~20GB compressed, ~100GB uncompressed
**Update**: Daily
**License**: CC BY-SA 3.0

#### Advantages:
- ✅ **Complete Database**: All articles, history, metadata
- ✅ **No Rate Limits**: Direct download
- ✅ **Custom Preprocessing**: Full control over data cleaning
- ✅ **Multiple Languages**: Available for all Wikipedia languages
- ✅ **Historical Data**: Access to previous versions

#### Usage:
```bash
# Download and process Wikipedia dump
python scripts/wikipedia_training_pipeline.py --source dumps --language en --max-articles 50000
```

#### Download URLs:
- **English**: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
- **German**: https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles-multistream.xml.bz2
- **French**: https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2

### 2. **HuggingFace Wikipedia Datasets** (Recommended for Quick Start)

**What**: Preprocessed Wikipedia datasets on HuggingFace
**Size**: Various (1GB-50GB)
**Update**: Periodic
**License**: CC BY-SA 3.0

#### Available Datasets:
- **`wikipedia`**: Multi-language Wikipedia articles
- **`wikipedia-corpus`**: Cleaned Wikipedia corpus
- **`wikipedia-sentences`**: Articles split into sentences

#### Advantages:
- ✅ **Preprocessed**: Clean, ready-to-use data
- ✅ **Multiple Formats**: Text, QA, embeddings
- ✅ **Easy Integration**: Direct HuggingFace loading
- ✅ **Memory Efficient**: Streaming support
- ✅ **Multiple Languages**: 11+ languages available

#### Usage:
```bash
# Quick start with HuggingFace dataset
python scripts/wikipedia_training_pipeline.py --source huggingface --language en --max-articles 10000
```

#### Python Code:
```python
from datasets import load_dataset

# Load Wikipedia dataset
dataset = load_dataset("wikipedia", language="en", date="20231201", split="train")
print(f"Loaded {len(dataset)} articles")

# Process for training
def format_for_training(example):
    return {"text": f"Title: {example['title']}\n\n{example['text']}"}

train_dataset = dataset.map(format_for_training)
```

### 3. **Wikipedia2Vec Embeddings** (For Knowledge Graph Training)

**What**: Pre-trained word embeddings from Wikipedia
**Size**: ~5GB
**Update**: Periodic
**License**: CC BY-SA 3.0

#### Advantages:
- ✅ **Pre-trained**: Ready-to-use embeddings
- ✅ **Knowledge Graph**: Entity and word embeddings
- ✅ **Multiple Languages**: Available for major languages
- ✅ **Efficient**: Optimized for knowledge tasks

#### Usage:
```python
import wikipedia2vec

# Load pre-trained embeddings
model = wikipedia2vec.Wikipedia2Vec.load("enwiki_20180420_100d.pkl")

# Get embeddings
word_vec = model.get_word_vector("artificial_intelligence")
entity_vec = model.get_entity_vector("Artificial_intelligence")
```

### 4. **Custom Wikipedia Scraping** (For Specific Content)

**What**: Targeted scraping of specific Wikipedia content
**Size**: Variable
**Update**: Real-time
**License**: CC BY-SA 3.0

#### Use Cases:
- Specific topics or categories
- Recent articles only
- Custom filtering requirements

#### Example Implementation:
```python
import requests
import json

def get_wikipedia_articles(categories, max_articles=1000):
    """Get articles from specific Wikipedia categories"""
    articles = []
    
    for category in categories:
        # Get category members
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "500",
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        for member in data["query"]["categorymembers"]:
            if member["ns"] == 0:  # Only main namespace articles
                # Get article content
                article_url = f"https://en.wikipedia.org/w/api.php"
                article_params = {
                    "action": "query",
                    "prop": "extracts",
                    "titles": member["title"],
                    "exintro": True,
                    "format": "json"
                }
                
                article_response = requests.get(article_url, params=article_params)
                article_data = article_response.json()
                
                # Extract content
                pages = article_data["query"]["pages"]
                for page_id, page_data in pages.items():
                    if "extract" in page_data:
                        articles.append({
                            "title": page_data["title"],
                            "content": page_data["extract"]
                        })
                
                if len(articles) >= max_articles:
                    break
    
    return articles
```

## 🚀 Quick Start Guide

### Option 1: HuggingFace (Recommended for Beginners)

```bash
# 1. Install dependencies
pip install datasets transformers torch

# 2. Run training pipeline
python scripts/wikipedia_training_pipeline.py --source huggingface --max-articles 10000

# 3. Check results
ls -la wikipedia_trained/
cat wikipedia_trained/training_report.json
```

### Option 2: Official Dumps (For Complete Training)

```bash
# 1. Download Wikipedia dump (20GB)
python scripts/wikipedia_training_pipeline.py --source dumps --max-articles 50000

# 2. Monitor progress
tail -f wikipedia_dumps/download.log

# 3. Train model
python scripts/wikipedia_training_pipeline.py --source dumps --model gpt2-medium
```

### Option 3: Custom Implementation

```python
# 1. Load Wikipedia dataset
from datasets import load_dataset
dataset = load_dataset("wikipedia", language="en", split="train[:10000]")

# 2. Prepare for training
def prepare_text(example):
    return {"text": f"Title: {example['title']}\n\n{example['text']}"}

train_dataset = dataset.map(prepare_text)

# 3. Train model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Setup training
training_args = TrainingArguments(
    output_dir="./wikipedia_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()
```

## 📊 Data Source Comparison

| Source | Size | Speed | Completeness | Ease of Use | Best For |
|--------|------|-------|--------------|-------------|----------|
| **MediaWiki API** | Rate limited | Slow | Partial | Easy | Specific queries |
| **Official Dumps** | ~100GB | Fast | Complete | Medium | Full training |
| **HuggingFace** | 1-50GB | Fast | High | Very Easy | Quick start |
| **Wikipedia2Vec** | ~5GB | Fast | High | Easy | Embeddings |
| **Custom Scraping** | Variable | Medium | Variable | Hard | Specific needs |

## 🎯 Recommendations by Use Case

### For Brain Simulation Training:
1. **Start with HuggingFace**: Quick validation and prototyping
2. **Move to Official Dumps**: For production training
3. **Combine with Wikipedia2Vec**: For knowledge graph integration

### For Research Projects:
1. **HuggingFace Datasets**: For rapid experimentation
2. **Custom Scraping**: For domain-specific content
3. **Official Dumps**: For comprehensive analysis

### For Production Systems:
1. **Official Dumps**: For complete training
2. **HuggingFace**: For fine-tuning
3. **Wikipedia2Vec**: For knowledge enhancement

## 🔧 Advanced Configuration

### Memory Optimization:
```python
# For large datasets
from datasets import load_dataset

# Stream dataset to avoid memory issues
dataset = load_dataset("wikipedia", language="en", streaming=True)
for batch in dataset.iter(batch_size=1000):
    # Process batch
    pass
```

### Multi-GPU Training:
```python
# Distributed training
training_args = TrainingArguments(
    output_dir="./wikipedia_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)
```

### Custom Preprocessing:
```python
def custom_cleaner(text):
    """Custom Wikipedia content cleaning"""
    import re
    
    # Remove Wikipedia markup
    text = re.sub(r'\[\[([^|\]]*?)\]\]', r'\1', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    
    return text.strip()

# Apply custom cleaning
dataset = dataset.map(lambda x: {"text": custom_cleaner(x["text"])})
```

## 📈 Performance Benchmarks

### Training Times (RTX 4090):
- **10K articles**: ~2 hours
- **100K articles**: ~20 hours
- **1M articles**: ~200 hours

### Memory Requirements:
- **GPT-2 Small**: 4GB VRAM
- **GPT-2 Medium**: 8GB VRAM
- **GPT-2 Large**: 16GB VRAM

### Dataset Sizes:
- **10K articles**: ~500MB
- **100K articles**: ~5GB
- **1M articles**: ~50GB

## 🎉 Success Stories

### Example Training Results:
```json
{
  "success": true,
  "model_path": "./wikipedia_trained",
  "training_loss": 2.34,
  "total_steps": 15000,
  "total_samples": 10000,
  "training_time": "2:15:30",
  "timestamp": "2025-01-20T10:30:00"
}
```

## 🚨 Important Notes

### Legal Considerations:
- Wikipedia content is licensed under CC BY-SA 3.0
- Attribution required for commercial use
- Share-alike clause applies to derivatives

### Technical Considerations:
- Wikipedia dumps are large (20GB+ compressed)
- Processing requires significant computational resources
- Consider using cloud computing for large-scale training

### Quality Considerations:
- Wikipedia content varies in quality
- Consider filtering by article length, references, etc.
- Implement custom quality metrics for your use case

## 🔗 Additional Resources

- [Wikipedia Dumps Documentation](https://dumps.wikimedia.org/)
- [HuggingFace Wikipedia Datasets](https://huggingface.co/datasets?search=wikipedia)
- [Wikipedia2Vec Documentation](https://wikipedia2vec.github.io/)
- [MediaWiki API Documentation](https://www.mediawiki.org/wiki/API:Main_page)

---

**Next Steps**: Choose your preferred data source and run the training pipeline. For brain simulation, we recommend starting with HuggingFace datasets and then moving to official dumps for comprehensive training.
