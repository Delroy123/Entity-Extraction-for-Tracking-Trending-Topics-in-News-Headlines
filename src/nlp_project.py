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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))

    # 1. Bar Chart: Top 20 Entities
    top_entities = entity_counts.most_common(10)
    if top_entities:
        names, counts = zip(*top_entities)
        ax1.barh(names, counts, color='skyblue')
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top 10 Trending Entities')
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center')

    # 2. Pie Chart
    if type_counts:
        labels = type_counts.keys()
        sizes = type_counts.values()
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center')
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
    plt.show()

# --- 5. MAIN APPLICATION LOOP ---

def main():
    print("--- NLP TERM PROJECT: TRENDING TOPIC TRACKER ---")
    download_nltk_resources()
    
    # User Input
    print("\nDo you want to upload a dataset? (Enter filename path, or press ENTER for Sample Data)")
    user_path = input("Path to CSV/JSON: ").strip()
    
    headlines = load_data(user_path if user_path else None)
    
    if not headlines:
        sys.exit("No data to process.")

    # --- RESTORED LIMITER ---
    if len(headlines) > 5000:
        print(f"Dataset is large ({len(headlines)} items). Processing first 1000 for efficiency.")
        headlines = headlines[:1000]
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
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_items} items...")
            
    end_time = time.time()
    
    # --- METRICS ---
    total_time = end_time - start_time
    avg_time = total_time / len(headlines) if len(headlines) > 0 else 0
    
    print("\n" + "="*40)
    print("EVALUATION METRICS")
    print("="*40)
    print(f"Total Processing Time:    {total_time:.4f} seconds")
    print(f"Avg Time per Headline:    {avg_time:.4f} seconds")
    print(f"Total Entities Found:     {len(all_entities)}")
    print("="*40)

    print("Generating visualizations...")
    entity_counts = Counter(all_entities)
    type_counts = Counter(all_entity_types)
    
    plot_results(entity_counts, type_counts)
    generate_wordcloud(all_entities)

    print("Top 10 Entities:", entity_counts.most_common(10))


if __name__ == "__main__":
    main()