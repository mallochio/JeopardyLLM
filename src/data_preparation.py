import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jeopardy_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jeopardy_data(data_path):
    """Load Jeopardy data from TSV file."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    logger.info(f"Loaded {len(df)} Jeopardy clues")
    return df

def generate_prompt(row: pd.Series) -> str:
    """Format data according to Gemma's chat template."""
    return f"""<bos><start_of_turn>user
# Instructions
# You are Alex Trebek hosting the current season of Jeopardy! You will provide a clue (the 'answer' in Jeopardy terms), and a contestant will respond with the correct 'question'. 

Jeopardy Round : {row['round']}
Category : {row['category']}
Value : {row['clue_value']}
Air Date : {row['air_date']}
Comments : {row['comments']}
Notes : {row['notes']}
Question : {row['question']}

<end_of_turn> <start_of_turn>model Here is your Jeopardy clue: "{row['answer']}" <end_of_turn><eos>"""

def generate_knowledge_prompt(row: pd.Series) -> str:
    """Format data to focus on knowledge acquisition from jeopardy queries."""
    return f"""<bos><start_of_turn>user
Topic: {row['category']}
Question: {row['answer']}
<end_of_turn> <start_of_turn>model
The answer is: {row['question']}
<end_of_turn><eos>"""

def prepare_data(data_path, output_path, prompt_style="knowledge"):
    """
    Prepare Jeopardy data for training.
    
    Args:
        data_path: Path to the data directory
        output_path: Path to save formatted data
        prompt_style: Either "knowledge" for simple knowledge format or 
                     "roleplay" for the full Jeopardy experience format
    """
    data_path = Path(data_path)
    df = load_jeopardy_data(data_path / "extra_matches.tsv")
    
    logger.info(f"Formatting data with {prompt_style} prompt style")
    if prompt_style == "roleplay":
        df["text"] = df.apply(generate_prompt, axis=1)
    else:  # knowledge style
        df["text"] = df.apply(generate_knowledge_prompt, axis=1)
    
    logger.info("Splitting data into train/validation sets")
    split_index = int(len(df) * 0.9)
    df_shuf = df.sample(frac=1, random_state=42)
    train, valid = df_shuf[:split_index], df_shuf[split_index:]
    
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    train_path = output_path / "train.jsonl"
    valid_path = output_path / "valid.jsonl"
    
    train[["text"]].to_json(train_path, orient="records", lines=True)
    valid[["text"]].to_json(valid_path, orient="records", lines=True)
    
    logger.info(f"Saved {len(train)} training and {len(valid)} validation examples to {output_path}")
    
    return train, valid
