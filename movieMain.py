import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DATA LOADING & PREPROCESSING (WITH CACHING) ---
@st.cache_data
def load_and_clean_data():
    """Loads all CSVs and performs the initial merge/cleaning."""
    # Load original datasets
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")

    # Select useful columns and drop missing
    movies = movies[['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'vote_count']]
    movies = movies.dropna(subset=['id', 'title', 'overview'])

    # Convert movie id to numeric
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)

    # Extract genres
    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return " ".join([g['name'] for g in genre_list])
        except:
            return ""

    movies['genres_clean'] = movies['genres'].apply(extract_genres)

    # Clean links and merge
    links = links[['movieId', 'tmdbId']]
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    movies_merged = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
    movies_merged = movies_merged.drop_duplicates(subset='title').reset_index(drop=True)

    # Create combined features for TF-IDF
    movies_merged['combined_features'] = (
        movies_merged['overview'].fillna('') + " " +
        (movies_merged['genres_clean'].fillna('') + " ") * 3
    )
    
    return movies_merged, ratings

@st.cache_resource
def compute_engine(movies_merged, ratings):
    """Computes the similarity matrices once and stores them in memory."""
    # Content-Based (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Collaborative (User-Item)
    ratings_movies = pd.merge(ratings, movies_merged, on='movieId', how='inner')
    user_movie_matrix = ratings_movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    # Title mapping index
    indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()
    
    return cosine_sim, movie_similarity_df, indices

# Initialize Data
movies_merged, ratings_data = load_and_clean_data()
cosine_sim, movie_similarity_df, indices = compute_engine(movies_merged, ratings_data)

# --- 2. CORE RECOMMENDATION LOGIC ---

def recommend_content(title, top_n=10):
    if title not in indices: return pd.DataFrame()
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+30]
    recs = pd.DataFrame({'title': [movies_merged.iloc[i[0]]['title'] for i in sim_scores], 'model_score': [i[1] for i in sim_scores]})
    return pd.merge(recs, movies_merged, on='title').query('vote_count > 50').head(top_n)

def recommend_collaborative(title, top_n=10):
    if title not in movie_similarity_df.columns: return pd.DataFrame()
    sim_scores = movie_similarity_df[title].sort_values(ascending=False)[1:top_n+30]
    recs = pd.DataFrame({'title': sim_scores.index, 'model_score': sim_scores.values})
    return pd.merge(recs, movies_merged, on='title').query('vote_count > 50').head(top_n)

def hybrid_recommend(movie_title, top_n=10, alpha=0.5):
    # Logic from your original script
    if movie_title not in indices or movie_title not in movie_similarity_df.columns:
        return pd.DataFrame()
    
    # Content Part
    idx = indices[movie_title]
    sim_content = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:100]
    content_df = pd.DataFrame({'title': [movies_merged.iloc[i[0]]['title'] for i in sim_content], 'c_score': [i[1] for i in sim_content]})
    
    # Collaborative Part
    sim_collab = movie_similarity_df[movie_title].sort_values(ascending=False)[1:100]
    collab_df = pd.DataFrame({'title': sim_collab.index, 'col_score': sim_collab.values})
    
    # Merge and Normalize
    hybrid = pd.merge(content_df, collab_df, on='title', how='outer').fillna(0)
    if hybrid['c_score'].max() != 0: hybrid['c_score'] /= hybrid['c_score'].max()
    if hybrid['col_score'].max() != 0: hybrid['col_score'] /= hybrid['col_score'].max()
    
    hybrid['model_score'] = alpha * hybrid['c_score'] + (1 - alpha) * hybrid['col_score']
    result = pd.merge(hybrid, movies_merged, on='title').sort_values('model_score', ascending=False)
    return result.query('vote_count > 50').head(top_n)

# --- 3. FILTERS (Mood & Personality) ---
def apply_filters(df, mood, personality, hidden_gems):
    if mood != "None":
        mood_map = {"Happy": ["Comedy", "Family"], "Sad": ["Drama"], "Romantic": ["Romance"], 
                    "Excited": ["Action", "Adventure"], "Curious": ["Mystery"], "Scared": ["Horror"], "Relaxed": ["Music"]}
        pattern = "|".join(mood_map.get(mood, []))
        df = df[df['genres_clean'].str.contains(pattern, case=False, na=False)]
    
    if personality != "None":
        pers_map = {"Adventurer": ["Adventure"], "Thinker": ["Science Fiction", "Mystery"], "Fun Lover": ["Comedy"]}
        pattern = "|".join(pers_map.get(personality, []))
        df = df[df['genres_clean'].str.contains(pattern, case=False, na=False)]
        
    if hidden_gems:
        df = df[(df['vote_average'] >= 6.5) & (df['vote_count'] < 500)]
    
    return df

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Movie Rec System", layout="wide")
st.title("🎬 Movie Recommendation System")

# Sidebar Navigation replaces the main_menu() loop
menu = st.sidebar.selectbox("Main Menu", ["Start Recommendation", "Search Movie Titles", "View Features"])

if menu == "Start Recommendation":
    mode = st.selectbox("Choose Type", ["Hybrid", "Content-Based", "Collaborative", "Explore", "New User"])
    
    movie_title = None
    if mode != "New User":
        movie_title = st.selectbox("Select a Movie", movies_merged['title'].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        mood = st.selectbox("Mood", ["None", "Happy", "Sad", "Romantic", "Excited", "Curious", "Scared", "Relaxed"])
    with col2:
        pers = st.selectbox("Personality", ["None", "Adventurer", "Thinker", "Fun Lover", "Dreamer"])
    with col3:
        top_n = st.number_input("How many results?", 1, 20, 5)

    hidden_gems = st.checkbox("Show Hidden Gems Only")
    
    if st.button("Get Recommendations"):
        if mode == "Content-Based":
            res = recommend_content(movie_title, top_n)
        elif mode == "Collaborative":
            res = recommend_collaborative(movie_title, top_n)
        else:
            res = hybrid_recommend(movie_title, top_n)
            
        final_res = apply_filters(res, mood, pers, hidden_gems)
        
        if not final_res.empty:
            for _, row in final_res.iterrows():
                st.subheader(row['title'])
                st.caption(f"Genres: {row['genres_clean']} | Rating: {row['vote_average']}")
                st.write(row['overview'])
                st.divider()
        else:
            st.warning("No movies found matching those filters.")

elif menu == "Search Movie Titles":
    search_q = st.text_input("Search by keyword")
    if search_q:
        matches = movies_merged[movies_merged['title'].str.contains(search_q, case=False, na=False)]
        st.table(matches[['title', 'genres_clean', 'vote_average']].head(15))

elif menu == "View Features":
    st.info("This system uses TF-IDF for Content-Based and Cosine Similarity for Collaborative filtering.")
