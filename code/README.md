# Group2Phase2 NLP Project

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine.

### 2. Install Requirements
Navigate to the `code/` directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download the All_Beauty Dataset
- The `All_Beauty.json` file is required for running the scripts. 
- Download it from the [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) page:
  - Go to the "Category: All Beauty" section.
  - Download the JSON file and place it in the `code/` directory as `All_Beauty.json`.

### 4. Cache FLAN-T5 Models
To download and cache the FLAN-T5 models locally, run:

```bash
python cache_flan_models.py
```

This will download and cache the required models into the `hf_cache/` and `hf_cache_large/` directories.

### 5. Run the Main Scripts
- For lexicon-based models (Phase 1):
  ```bash
  python phase1_lexicon_models.py
  ```
- For machine learning recommender and LLM (Phase 2):
  ```bash
  python phase2_ml_recommender_llm.py
  ```
