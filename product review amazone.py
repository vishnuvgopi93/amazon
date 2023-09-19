import requests
from bs4 import BeautifulSoup as bs
import re
def scrape_amazon_reviews(product_url, num_pages=20):
    reviews_list = []

    for page_number in range(1, num_pages + 1):
        try:
            ip = []
            url = f"{product_url}&pageNumber={page_number}"
            response = requests.get(url)
            response.raise_for_status()

            soup = bs(response.content, "html.parser")
            reviews = soup.find_all("span", class_="a-size-base review-text review-text-content")

            if not reviews:
                break  # No more reviews found, exit the loop

            for review in reviews:
                ip.append(review.get_text(strip=True))

            reviews_list.extend(ip)
        except Exception as e:
            print(f"Error scraping page {page_number}: {e}")

    return reviews_list

# Example usage:
product_url = "https://www.amazon.in/Bajaj-Majesty-2800-TMCSS-28-Litre/product-reviews/B009P2KOSC/ref=cm_cr_othr_d_show_all_btm?ie=UTF8&filterByKeyword=made+in+china#reviews-filter-bar"
product_reviews = scrape_amazon_reviews(product_url, num_pages=20)

# Write reviews to a text file
with open("product_reviews.txt", "w", encoding='utf-8') as output:
    output.write("\n".join(product_reviews))

ip_rev_string = " ".join(product_reviews)

#import nltk
#from nltk.corpus import stopwords

# Removing unwanted symbols incase they exists
ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# Words that are contained in the reviews
ip_reviews_words = ip_rev_string.split(" ")

ip_reviews_words = ip_reviews_words[1:]

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf = True, ngram_range = (1, 1))
X = vectorizer.fit_transform(ip_reviews_words)

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')  # Download the stop words data (if not already downloaded)

stop_words = set(stopwords.words('english'))

'''
# You can also add custom stop words to the set if needed
custom_stop_words = ["Amazon", "echo", "time", "android", "phone", "device", "product", "day"]
stop_words.update(custom_stop_words)

with open("D:\\Data\\textmining\\stop.txt", "r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["Amazon", "echo", "time", "android", "phone", "device", "product", "day"])
'''
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs.
# Corpus level word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt


wordcloud_ip = WordCloud(background_color = 'White',
                      width = 1800,  height = 1400
                     ).generate(ip_rev_string)
plt.imshow(wordcloud_ip)

# Positive words # Choose the path for +ve words stored in system
with open("D:\positive-words.txt", "r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color = 'White',
                      width = 1800,
                      height = 1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# Negative word cloud
# Choose path for -ve words stored in system
with open("D:\\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color = 'black',
                      width = 1800,
                      height = 1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

#################################################################
# Joining all the reviews into single paragraph 
ip_rev_string = " ".join(product_reviews)

# Wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars as well as stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great', '9rt'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range = (2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width, stopwords = new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

