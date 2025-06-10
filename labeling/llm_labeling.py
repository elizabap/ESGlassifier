import pandas as pd
from ollama import Client
import time

client = Client(host='http://localhost:11434')

df = pd.read_csv('../data/base_data_train_G.csv')

governance_keywords = [
    'audit', 'authority', 'practice', 'bribery', 'code', 'compensation', 'competition', 'competitive', 'compliance', 
    'conflict of interest', 'control', 'corporate', 'corruption', 'crisis', 'culture', 'decision', 'due diligence', 
    'duty', 'ethic', 'governance', 'framework', 'issue', 'structure', 'guideline', 'integrity', 'internal', 'lead', 
    'legal', 'lobby', 'oversight', 'policy', 'politic', 'procedure', 'regulat', 'reporting', 'responsib', 'right', 
    'management', 'sanction', 'stake', 'standard', 'transparen', 'vot', 'whistleblower', 'accounting', 'accountable', 
    'accountant', 'accounted'
]


environmental_keywords = [
    'adaptation', 'quality', 'animal', 'atmospher', 'biomass', 'capture', 'ch4', 'climat', 'co2', 'coastal',
    'concentration', 'conservation', 'consumption', 'degree', 'depletion', 'dioxide', 'diversity', 'drought', 
    'ecolog', 'ecosystem', 'ecosystems', 'emission', 'emissions', 'energy', 'environment', 'environmental', 
    'flood', 'footprint', 'forest', 'fossil', 'fuel', 'fuels', 'gas', 'gases', 'ghg', 'global warming', 'green', 
    'greenhouse', 'hydrogen', 'impacts', 'land use', 'methane', 'mitigation', 'n2o', 'nature', 'nitrogen', 'ocean', 
    'ozone', 'plant', 'pollution', 'rainfall', 'renewable', 'resource', 'seasonal', 'sediment', 'snow', 'soil', 
    'solar', 'sources', 'sustainab', 'temperature', 'thermal', 'trees', 'tropical', 'waste', 'water'
]

social_keywords = [
    'age', 'cultur', 'race', 'access to', 'accessibility', 'accident', 'accountability', 'awareness', 'behaviour', 
    'charity', 'civil', 'code of conduct', 'communit', 'community', 'consumer protection', 'cyber security', 
    'data privacy', 'data protection', 'data security', 'demographic', 'disability', 'disable', 'discrimination', 
    'divers', 'donation', 'education', 'emotion', 'employee benefit', 'employee development', 'employment benefit', 
    'empower', 'equal', 'esg', 'ethic', 'ethnic', 'fairness', 'family', 'female', 'financial protection', 'gap', 
    'gender', 'health', 'human', 'inclus', 'information security', 'injury', 'leave', 'lgbt', 'peace', 'privacy', 
    'mental', 'pension', 'well-being', 'parity', 'pay equity', 'benefit', 'philanthrop', 'poverty', 'product', 
    'promotion', 'quality of life', 'religion', 'respectful', 'quality', 'product safety', 'respecting', 
    'retirement benefit', 'safety', 'supply chain', 'salary', 'social', 'society', 'transparency', 'supportive', 
    'talent', 'volunteer', 'wage', 'welfare', 'well-being', 'wellbeing', 'wellness', 'women', 'workforce', 
    'working conditions'
]


def make_prompt(sentence):
    keywords_list = ", ".join(governance_keywords)
    return f"""You are a language model trained to detect whether a sentence relates to *governance topics* within ESG. 
    ESG (Environmental, Social, and Governance) refers to a set of criteria used to evaluate a company's ethical impact and sustainability practices in these three areas.


Classify the following sentence as either:
1 → Governance-related  
0 → Not governance-related

Governance-related sentences typically involve one or more of these concepts: {keywords_list}.

Sentence: "{sentence}"


Reply with only a single digit: 1 (if governance-related) or 0 (if not)."""

def classify_sentence(sentence, model="mistral"):
    prompt = make_prompt(sentence)
    try:
        response = client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        label = response['message']['content'].strip()
        return int(label) if label in ['0', '1'] else None
    except Exception as e:
        print(f"Error classifying sentence: {e}")
        return None

df['gov'] = df['sentence'].apply(lambda x: classify_sentence(x))

df.to_csv('ollama_G.csv', index=False)

print("Binary classification complete!")
