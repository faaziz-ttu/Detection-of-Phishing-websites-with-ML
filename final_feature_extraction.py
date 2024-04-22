import pandas as pd
import re
import random
import datetime
import whois

def using_ip_address(url):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    if re.search(ip_pattern, url):
        return 0  # Phishing 
    else:
        return 1  # Legitimate

def classify_url_length(length):
    if length < 54:
        return 1  # Legitimate
    elif 54 <= length <= 75:
        return 0  # Suspicious URLs are classified as Phishing
    else:
        return 0

def contains_short_url(url):
    short_url_services = ['tinyurl', 'bitly', 'rebrandly', 'bl\.ink', 'short\.io', 't\.ly', 'buff\.ly']
    pattern = r'\b(?:' + '|'.join(short_url_services) + r')\b'
    if re.search(pattern, url):
        return 0  # Phishing
    else:
        return 1  # Legitimate

def contains_at_symbol(url):
    if '@' in url:
        return 0  # Phishing
    else:
        return 1  # Legitimate

def last_occurrence_position(url):
    last_occurrence = url.rfind("//")
    if last_occurrence > 7:
        return 0  # Phishing
    else:
        return 1  # Legitimate

def contains_hyphen_in_domain(url):
    domain = url.split('//')[-1].split('/')[0]
    if '-' in domain:
        return 0  # Phishing
    else:
        return 1  # Legitimate

def dots_in_domain(url):
    domain = url.split('//')[-1].split('/')[0]
    dot_count = domain.count('.')
    if dot_count == 2:
        return 1  # Legitimate
    else:
        return 0  # Phishing

def nums_in_url(url):
    domain_name = url.split('//')[-1].split('/')[0]
    num_in_domain_name = sum(c.isdigit() for c in domain_name)
    num_in_url = sum(c.isdigit() for c in url)
    if num_in_domain_name > 0 or num_in_url > 10:
        return 0 # Phishing
    else:
        return 1  # Legitimate

def https_classification(url):
    if url.startswith("http://"):
        return 0  # Phishing
    elif url.startswith("https://"):
        return 1  # Legitimate
    else:
        return 0 # Phishing      


def url_age(url):
    try:
        details = whois.whois(url)
        if details.creation_date:
            if isinstance(details.creation_date, list):
                created_date = min(details.creation_date)
            else:
                created_date = details.creation_date
            
            today = datetime.datetime.today()
            age_in_months = (today.year - created_date.year) * 12 + today.month - created_date.month
            
            if age_in_months >= 6:
                return 0  # Phishing
            else:
                return 1  # Legitimate
        else:
            print(f"No creation date found for {url}")
            return 0  # Assuming no creation date indicates a legitimate site
    except Exception as e:
        print(f"Error fetching WHOIS details for {url}: {e}")
        return 0  # Assuming error fetching WHOIS details indicates a legitimate site


# Read the CSV file
df = pd.read_csv('new_dataset.csv')

# Add using_ip_address column to the dataframe
df['using_ip_address'] = df['URL'].apply(using_ip_address)

# Add URL_length column to the dataframe
df['URL_length'] = df['URL'].apply(len)
    
# Add URL_length_classification column based on the URL length
df['URL_length_classification'] = df['URL_length'].apply(classify_url_length)

df['containing_shortUrl'] = df['URL'].apply(contains_short_url)

df['containing_at_symbol'] = df['URL'].apply(contains_at_symbol)

df['last_occurrence_position'] = df['URL'].apply(last_occurrence_position)

df['containing_hyphen_in_domain'] = df['URL'].apply(contains_hyphen_in_domain)

df['dots_in_domain'] = df['URL'].apply(dots_in_domain)

df['nums_in_url'] = df['URL'].apply(nums_in_url)

df['https_classification'] = df['URL'].apply(https_classification)

#df['url_age'] = df['URL'].apply(url_age)

# Save the dataframe back to the same CSV file, overwriting the original file
df.to_csv('new_dataset.csv', index=False)



random_lines = pd.read_csv('new_dataset.csv', header=0, skiprows=lambda i: i>0 and random.random() > 0.5, nrows=15)

print(random_lines)