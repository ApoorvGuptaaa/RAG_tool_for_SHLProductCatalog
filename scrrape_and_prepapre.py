import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def scrape_shl_catalog():
    url = 'https://www.shl.com/solutions/products/product-catalog/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    assessments = []

    cards = soup.find_all('a', class_='product-card')  # Card selector
    for card in cards:
        name = card.find('div', class_='product-title')
        duration = card.find('div', class_='product-duration')
        details = card.find('div', class_='product-description')

        if name:
            title = name.text.strip()
            link = card['href'] if card.has_attr('href') else ''
            description = details.text.strip() if details else ''
            duration_text = duration.text.strip() if duration else ''

            assessments.append({
                'name': title,
                'url': 'https://www.shl.com' + link,
                'description': description,
                'duration': duration_text,
                'remote_testing': "Yes",  # Placeholder (you can refine if real data found)
                'adaptive_irt': "Yes",    # Placeholder
                'test_type': "Psychometric",  # Placeholder
            })

    return assessments

def generate_embeddings(assessments):
    for assess in assessments:
        assess['embedding'] = model.encode(assess['description']).tolist()
    return assessments

if __name__ == '__main__':
    print("Scraping SHL catalog...")
    data = scrape_shl_catalog()
    print(f"Scraped {len(data)} assessments.")

    print("Generating embeddings...")
    data = generate_embeddings(data)

    with open('assessments_with_embeddings.json', 'w') as f:
        json.dump(data, f)

    print("Done. Saved to assessments_with_embeddings.json")
