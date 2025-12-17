import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import time
import re
import sys
import pickle
import os

# --- SYSTEM VARIABLE
MAXIMUM_ROWS = 1000
RESULT_DIRECTORY = 'result'

# --- STORAGE UTILITIES ---

def get_dataset_name(filepath):
    if not filepath:
        return "sample"
    return os.path.splitext(os.path.basename(filepath))[0]

def get_dataset_folder(dataset_name):
    return os.path.join(os.getcwd(), dataset_name)

def save_results(folder, data):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "results.pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"Results saved to folder: {folder}")

def load_results(folder):
    with open(os.path.join(folder, "results.pkl"), "rb") as f:
        return pickle.load(f)


# --- 1. CONFIGURATION & SETUP ---
def download_nltk_resources():
    """Ensures necessary NLTK models are downloaded."""
    resources = [
        'punkt', 
        'punkt_tab',
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng', 
        'maxent_ne_chunker', 
        'words', 
        'stopwords',
        'maxent_ne_chunker_tab' 
    ]
    print("Checking NLTK resources...")
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {res}. Error: {e}")
    print("NLTK resources check complete.\n")

# --- 2. IMPROVED PREPROCESSING (Fixes "Mr." and URLs) ---

def preprocess_text(text):
    """
    Cleans text by removing:
    1. URLs (e.g., shopping.com)
    2. Honorifics (Mr., Mrs., Dr.)
    3. Non-alphanumeric symbols (keeping spaces)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs (http://... or something.com)
    text = re.sub(r'http\S+|www\.\S+|\S+\.com\S*', '', text)
    
    # 2. Remove specific Honorifics (Mr., Mrs., Dr.) - Case sensitive to protect names
    # We replace them with an empty string
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof)\b\.?', '', text, flags=re.IGNORECASE)
    
    # 3. Remove non-alphanumeric (keep spaces, periods for sentence structure)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    
    return text.strip()

def get_entities(text):
    """
    Methodology Step: NER using ne_chunk() with STOPWORD FILTERING
    """
    try:
        # Load stopwords once
        stop_words = set(stopwords.words('english'))
        # Add custom noise words often found in news
        custom_stops = {'said', 'says', 'would', 'could', 'also', 'new', 'one', 'two'}
        stop_words.update(custom_stops)

        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        chunks = ne_chunk(tags)
        
        entities = []
        
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                if chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                    # Join multi-word entities
                    entity_name = ' '.join(c[0] for c in chunk)
                    
                    # --- NEW FILTERING LOGIC ---
                    # 1. Ignore if the entity is just a stopword (e.g., "The")
                    if entity_name.lower() in stop_words:
                        continue
                    
                    # 2. Ignore short garbage (e.g., single letters or just "Mr")
                    if len(entity_name) < 3:
                        continue
                        
                    entities.append((entity_name, chunk.label()))
        
        return entities
    except Exception as e:
        return []

# --- 3. DATA LOADING ---

def load_data(filepath=None):
    if filepath:
        try:
            print(f"Loading data from {filepath}...")
            if filepath.endswith('.json'):
                df = pd.read_json(filepath, lines=True)
            else:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin1')
            
            possible_cols = ['headline', 'text', 'title', 'short_description']
            text_col = next((col for col in possible_cols if col in df.columns), None)
            
            if text_col:
                print(f"Successfully loaded. Using column: '{text_col}'")
                return df[text_col].dropna().tolist()
            else:
                print("Error: Could not find a 'headline' or 'text' column.")
                return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    else:
        print("No file provided. Using SAMPLE DATA.")
        return ["Sample headline"]

# --- 4. VISUALIZATION ---

def plot_results(entity_counts, type_counts):
    if not entity_counts:
        print("No entities found to visualize.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Trending Topics Analysis (Sampled Data)', fontsize=16)

    # 1. Bar Chart: Top 20 Entities
    top_entities = entity_counts.most_common(20)
    if top_entities:
        names, counts = zip(*top_entities)
        ax1.barh(names, counts, color='skyblue')
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top 20 Trending Entities')
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center')

    # 2. Pie Chart
    if type_counts:
        labels = type_counts.keys()
        sizes = type_counts.values()
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax2.set_title('Distribution of Entity Types')
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center')

    plt.tight_layout()
    plt.show()

def generate_wordcloud(entity_list):
    if not entity_list:
        return
    
    most_common = Counter(entity_list).most_common(2000)
    text_dict = dict(most_common)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(text_dict)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Trending Topics Word Cloud")
    plt.show()

# --- 5. MAIN APPLICATION LOOP ---

def show_results(results):
    total_time = results["total_time"]
    avg_time = results["avg_time"]
    total_entities = results["total_entities"]

    print("\n" + "="*40)
    print("EVALUATION METRICS")
    print("="*40)
    print(f"Total Processing Time:    {total_time:.4f} seconds")
    print(f"Avg Time per Headline:    {avg_time:.4f} seconds")
    print(f"Total Entities Found:     {total_entities}")
    print("="*40)

    print("Generating visualizations...")

    entity_counts = results["entity_counts"]
    type_counts = results["type_counts"]
    all_entities = results["all_entities"]

    plot_results(entity_counts, type_counts)
    generate_wordcloud(all_entities)

def main():
    print("--- NLP TERM PROJECT: TRENDING TOPIC TRACKER ---")
    download_nltk_resources()
    
    # User Input
    print("\nDo you want to upload a dataset? (Enter filename path, or press ENTER for Sample Data)")
    user_path = input("Path to CSV/JSON: ").strip()
    
    dataset_name = get_dataset_name(user_path)
    dataset_folder = get_dataset_folder(RESULT_DIRECTORY + '/' + dataset_name)

    # Check if dataset already processed
    if os.path.exists(dataset_folder):
        print(f"\nDataset '{dataset_name}' has been processed before.")
        choice = input("Are you sure you want to do it all over again? (y/n): ").strip().lower()

        if choice != 'y':
            print("Loading previously saved results...")
            saved = load_results(dataset_folder)
            show_results(saved)
            return
        else:
            print("Retraining and overwriting previous results...\n")

    headlines = load_data(user_path if user_path else None)

    
    if not headlines:
        sys.exit("No data to process.")

    # --- RESTORED LIMITER ---
    if len(headlines) > MAXIMUM_ROWS:
        print(f"Dataset is large ({len(headlines)} items). Processing first {MAXIMUM_ROWS} for efficiency.")
        headlines = headlines[:MAXIMUM_ROWS]
    else:
        print(f"Processing {len(headlines)} headlines...")
    
    all_entities = []
    all_entity_types = []
    
    start_time = time.time()
    
    # --- PROCESSING PIPELINE ---
    total_items = len(headlines)
    
    for i, text in enumerate(headlines):
        clean = preprocess_text(text)
        extracted = get_entities(clean)
        
        for name, label in extracted:
            all_entities.append(name)
            all_entity_types.append(label)
            
        if (i + 1) % int(MAXIMUM_ROWS/10) == 0:
            print(f"Processed {i + 1}/{total_items} items...")
            
    end_time = time.time()
    
    # --- METRICS ---
    total_time = end_time - start_time
    avg_time = total_time / len(headlines) if len(headlines) > 0 else 0
    
    entity_counts = Counter(all_entities)
    type_counts = Counter(all_entity_types)
    results = {
        "entity_counts": entity_counts,
        "type_counts": type_counts,
        "all_entities": all_entities,
        "total_time": total_time,
        "avg_time": avg_time,
        "total_entities": len(all_entities)
    }
    
    show_results(results)
    # --- SAVE RESULTS ---

    save_results(dataset_folder, results)


if __name__ == "__main__":
    main()