#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for the project, you'll have to import the libraries you'll need, you can find a list of the ones available in this workspace in the requirements.txt file in this workspace. 

# # Synthetic Data Generation

# ### Criteria 1 - Generating Real Estate Listings with an LLM
# The submission must demonstrate using a Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install lancedb')


# In[2]:


# import necessary libraries
import openai
import os
import pandas as pd
import numpy as np
import lancedb
import json

# OpenAI key here.
openai.api_key = "my key"


# In[3]:


# Define a function to generate real estate listings
def generate_real_estate_listings():
    prompt = """
    Generate 3 diverse and realistic real estate listings containing facts about the real estate, such as:
    - Location (city, suburb)
    - Type of the real estate (apartment, house, townhouse, etc.)
    - Sale price
    - Year built
    - Size of the real estate (square meters)
    - Number of bedrooms
    - Number of bathrooms
    - Garage space (number of cars)
    - Proximity to public transport, schools, shops
    - neighborhood vibes
    - architectural styles
    - other unique characteristics of the property, etc.

    Each listing should be unique and realistic.
    """

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are an expert in generating real estate listings."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    listings = response.choices[0].message['content']
    return listings

# Generate the listings and print them
listings = generate_real_estate_listings()
print(listings)


# In[52]:


def parse_listings(listings):
    # Example parsing logic; adapt based on actual format
    listings_list = listings.strip().split("\n\n")  # Assuming each listing is separated by double newline
    parsed_listings = []
    
    for listing in listings_list:
        listing_dict = {}
        for line in listing.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                # Remove numbering and leading hyphens from keys
                key = key.split(".")[-1].strip().lstrip("-").strip()
                listing_dict[key] = value.strip()
        parsed_listings.append(listing_dict)
    
    return parsed_listings

def create_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Generate the listings
listings = generate_real_estate_listings()

# Parse the listings
listings_data = parse_listings(listings)

# Create the listings data in the specified format
listings_with_vectors = []
for listing in listings_data:
    listing_text = ' '.join([f"{key}: {value}" for key, value in listing.items()])
    vector = create_embeddings([listing_text])[0]
    listing['vector'] = vector
    listings_with_vectors.append(listing)

listings_with_vectors


# # Semantic Search

# ### Criteria 2- Creating a Vector Database and Storing Listings
# The project must demonstrate the creation of a vector database and successfully storing real estate listing embeddings within it. The database should effectively store and organize the embeddings generated from the LLM-created listings.

# In[63]:


# Create LanceDB instance and database
data = listings_with_vectors
db = lancedb.connect("~/.lancedb")
tbl = db.create_table("my_listings", data, mode="overwrite")
len(tbl)


# ### Criteria 3 - Semantic Search of Listings Based on Buyer Preferences
# The application must include a functionality where listings are semantically searched based on given buyer preferences. The search should return listings that closely match the input preferences

# In[64]:


tbl.to_pandas().head(1)


# In[65]:


def create_prompt(query, context):
    limit = 3750

    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(context)):
        if len("\n\n---\n\n".join(context.text[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context.text[:i-1]) +
                prompt_end
            )
            break
        elif i == len(context)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context.text) +
                prompt_end
            )
    return prompt


# In[66]:


def complete(prompt):
    res = openai.Completion.create(
        model='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res.choices[0].text

# check that it works
query = "provide a list of properties in New York"
complete(query)


# In[67]:


query = ("Provide me with a list of properties in New York with 3 bedrooms.")


# In[68]:


emb_query = create_embeddings(query)
emb_query


# In[70]:


context = tbl.search(emb_query).select(["Location","Type", "Sale Price", "Year Built", "Size", "Bedrooms", "Bathrooms", "Garage Space", "Proximity", "Neighborhood Vibes", "Architectural Style", "Unique Characteristics" "vector"]).limit(2).to_pandas()


# In[49]:


prompt = create_prompt(query, context)
complete(prompt)


# In[ ]:




