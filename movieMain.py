import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DATA LOADING & PREPROCESSING (ORIGINAL LOGIC) ---
@st.cache_data
def load_data():
    """All original data cleaning from your script"""
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")

    # Select and clean
    movies = movies[['id', 'title', 'overview', 'genres', 'release_date', 'vote_average', 'vote_count']]
    movies = movies.dropna(subset=['id', 'title', 'overview'])
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id']).astype({'id': int})

    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return " ".join([g['name'] for g in genre_list])
        except: return ""

    movies['genres_clean'] = movies['genres'].apply(extract_genres)
    
    # Merge datasets
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce').dropna().astype(int)
    merged = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
    merged = merged.drop_duplicates(subset='title').reset_index(drop=True)
    
    # Feature combination for TF-IDF
    merged['combined_features'] = (
        merged['overview'].fillna('') + " " + 
        (merged['genres_clean'].fillna('') + " ") * 3
    )
    return merged, ratings

@st.cache_resource
def compute_matrices(movies_merged, ratings):
    """Original similarity calculations"""
    # Content Similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_merged['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Collaborative Similarity
    ratings_movies = pd.merge(ratings, movies_merged, on='movieId', how='inner')
    user_movie_matrix = ratings_movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_sim = cosine_similarity(user_movie_matrix.T)
    movie_sim_df = pd.DataFrame(movie_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    
    indices = pd.Series(movies_merged.index, index=movies_merged['title']).drop_duplicates()
    return cosine_sim, movie_sim_df, indices

# Initialize State
movies_merged, ratings_data = load_data()
cosine_sim, movie_similarity_df, indices = compute_matrices(movies_merged, ratings_data)

# --- 2. FULL ORIGINAL FEATURE LOGIC ---

def get_recommendations(movie_title=None, mode="Hybrid", top_n=10, mood=None, personality=None, hidden_gems=False, alpha=0.5):
    """Comprehensive recommendation engine including all your original modes"""
    results = pd.DataFrame()

    if mode == "New User":
        results = movies_merged.copy()
        if personality != "None":
            results = apply_personality_filter(results, personality)
        if mood != "None":
            results = apply_mood_filter(results, mood)
        results = results.sort_values(by=['vote_average', 'vote_count'], ascending=False).query('vote_count > 100')
        results['explanation'] = "Recommended for new users based on mood/personality."

    elif mode == "Explore":
        results = hybrid_recommend(movie_title, top_n=20, alpha=0.5)
        results = results.iloc[5:5+top_n] if len(results) > 5 else results
        results['explanation'] = "Explore mode: Less obvious but relevant choices."

    elif mode == "Content-Based":
        results = recommend_content(movie_title, top_n)
        results['explanation'] = results.apply(lambda r: f"Similar storyline/genres: {r['genres_clean']}.", axis=1)

    elif mode == "Collaborative":
        results = recommend_collaborative(movie_title, top_n)
        results['explanation'] = "Based on similar user behavior."

    else: # Hybrid
        results = hybrid_recommend(movie_title, top_n, alpha)
        results['explanation'] = "Combined content and collaborative analysis."

    # Applying standard filters
    if mood != "None" and mode != "New User":
        results = apply_mood_filter(results, mood)
    if personality != "None" and mode != "New User":
        results = apply_personality_filter(results, personality)
    if hidden_gems:
        results = results[(results['vote_average'] >= 6.5) & (results['vote_count'] < 500)]
        
    return results.head(top_n)

# Original Support Functions
def recommend_content(title, n):
    idx = indices[title]
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+30]
    df = pd.DataFrame({'title': [movies_merged.iloc[i[0]]['title'] for i in scores], 'model_score': [i[1] for i in scores]})
    return pd.merge(df, movies_merged, on='title').query('vote_count > 50')

def recommend_collaborative(title, n):
    scores = movie_similarity_df[title].sort_values(ascending=False)[1:n+30]
    df = pd.DataFrame({'title': scores.index, 'model_score': scores.values})
    return pd.merge(df, movies_merged, on='title').query('vote_count > 50')

def hybrid_recommend(title, n, alpha):
    if title not in indices or title not in movie_similarity_df.columns: return pd.DataFrame()
    c_recs = recommend_content(title, 100).rename(columns={'model_score': 'c_score'})
    col_recs = recommend_collaborative(title, 100).rename(columns={'model_score': 'col_score'})
    hybrid = pd.merge(c_recs[['title', 'c_score']], col_recs[['title', 'col_score']], on='title', how='outer').fillna(0)
    hybrid['model_score'] = (alpha * (hybrid['c_score']/hybrid['c_score'].max())) + ((1-alpha) * (hybrid['col_score']/hybrid['col_score'].max()))
    return pd.merge(hybrid, movies_merged, on='title').sort_values('model_score', ascending=False)

def apply_mood_filter(df, mood):
    mood_map = {"Happy": ["Comedy", "Family"], "Sad": ["Drama"], "Romantic": ["Romance"], "Excited": ["Action"], "Curious": ["Mystery"], "Scared": ["Horror"], "Relaxed": ["Music"]}
    pattern = "|".join(mood_map.get(mood, []))
    return df[df['genres_clean'].str.contains(pattern, case=False, na=False)]

def apply_personality_filter(df, pers):
    pers_map = {"Adventurer": ["Adventure"], "Romantic": ["Romance"], "Thinker": ["Science Fiction"], "Fun Lover": ["Comedy"], "Dreamer": ["Fantasy"]}
    pattern = "|".join(pers_map.get(pers, []))
    return df[df['genres_clean'].str.contains(pattern, case=False, na=False)]

# --- 3. STREAMLIT INTERFACE (THE FRONT END) ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.sidebar.title("🎬 Menu")
page = st.sidebar.radio("Navigate", ["Recommendations", "Search Titles", "System Features", "Method Comparison"])

if page == "Recommendations":
    st.header("🎬 Movie Recommendations")
    mode = st.selectbox("Mode", ["Hybrid", "Content-Based", "Collaborative", "Explore", "New User"])
    movie_title = st.selectbox("Select Movie", movies_merged['title'].unique()) if mode != "New User" else None
    
    col1, col2, col3 = st.columns(3)
    mood = col1.selectbox("Mood", ["None", "Happy", "Sad", "Romantic", "Excited", "Curious", "Scared", "Relaxed"])
    pers = col2.selectbox("Personality", ["None", "Adventurer", "Romantic", "Thinker", "Fun Lover", "Dreamer"])
    top_n = col3.slider("Count", 1, 10, 5)
    gems = st.checkbox("Hidden Gems Only")

    if st.button("Generate"):
        res = get_recommendations(movie_title, mode, top_n, mood, pers, gems)
        if not res.empty:
            for _, row in res.iterrows():
                with st.expander(f"🎥 {row['title']} (Score: {row.get('model_score', 'N/A')})"):
                    st.write(f"**Genres:** {row['genres_clean']} | **Rating:** {row['vote_average']}")
                    st.write(f"**Why?** {row['explanation']}")
                    st.write(f"**Plot:** {row['overview']}")
        else: st.error("No results found.")

elif page == "Search Titles":
    query = st.text_input("Keyword:")
    if query:
        st.table(movies_merged[movies_merged['title'].str.contains(query, case=False)].head(10)[['title', 'genres_clean']])

elif page == "System Features":
    st.header("✨ Full System Features")
    features = ["Content-Based", "Collaborative Filtering", "Hybrid Engine", "Explore Mode", "New User Logic", "Mood Filter", "Personality Filter", "Hidden Gems", "Explainable Recs", "Method Comparison"]
    for i, f in enumerate(features, 1): st.write(f"{i}. **{f}**")

elif page == "Method Comparison":
    title = st.selectbox("Compare for:", movies_merged['title'].unique())
    if st.button("Run Comparison"):
        c1, c2, c3 = st.columns(3)
        c1.write("**Content**"); c1.table(recommend_content(title, 5)[['title']])
        c2.write("**Collaborative**"); c2.table(recommend_collaborative(title, 5)[['title']])
        c3.write("**Hybrid**"); c3.table(hybrid_recommend(title, 5, 0.5)[['title']])
