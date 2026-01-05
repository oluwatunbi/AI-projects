import os 
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()


client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = '2024-12-01-preview'           
    )

model = "text-embedding-ada-002"
SIMILARITY_THRESHOLD = 0.75
dataset_name = './embedding_index_3m.json'

# load the dataset into a dataframe
def load_dataset(dataset: str) -> pd.core.frame.DataFrame:
    # load video data section index 
    pd_vector = pd.read_json(dataset)
    return pd_vector.drop(columns = ['text'], errors = 'ignore').fillna("")

# write the similarity equation
def counsine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# create a function to perform vector search 
def get_video(query: str, dataset: pd.core.frame.DataFrame, rows: int)-> pd.core.frame.DataFrame:
    # create a copy of the dtatset
    video_dataset = dataset.copy()
    
    # get embedding fro the query
    query_embedding = client.embeddings.create(input = query, model = model).data[0].embedding
    
    # create a new column for similarity score
    video_dataset['similarity_score'] = video_dataset['ada_v2'].apply(
        lambda x: counsine_similarity(np.array(query_embedding), np.array(x))
    )
    
    # filter similarity videos above threshold
    filtered_videos = video_dataset['similarity_score'] >= SIMILARITY_THRESHOLD
    video_vector = video_dataset[filtered_videos].copy()
    
    #sort the video with similarity
    video_vector = video_vector.sort_values(by= 'similarity_score', ascending = False).head(rows)
    
    return video_vector.head(rows)

# create a function to print out the url
def display_results(video: pd.core.frame.DataFrame, query: str):
    def get_ut_url(video_id: str, seconds: int) -> str:
        return f"https://youtube/{video_id}?t={seconds}"
    
    print(f"\nVideos similar to '{query}': ")
    # loop through the dataframe and print the url
    for _, row in video.iterrows():
        youtube_url = get_ut_url(row['videoId'], row['seconds'])
        print(f" {row['title']}")
        print(f" summary: {' '.join(row['summary'].split()[:15])}...")
        print(f" Youtube URL: {youtube_url}")
        print(f" Similarity: {row['similarity_score']}")
        print(f" Speaker: {row['speaker']}")
    
# call the function get query from user
pd_vector = load_dataset(dataset_name)

while True:
    query = input('enter your query: ')
    if query == 'exit':
        break
    
    video = get_video(query, pd_vector, 5)
    display_results(video, query)
