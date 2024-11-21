from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset
df = pd.read_csv("updated_genre_data.csv")

# Define genre columns in the dataset
genre_columns = [
    'Genre.Adventure', 'Genre.Fantasy', 'Genre.Animation', 'Genre.Drama', 'Genre.Horror', 
    'Genre.Action', 'Genre.Comedy', 'Genre.History', 'Genre.Western', 'Genre.Thriller', 
    'Genre.Crime', 'Genre.Documentary', 'Genre.Science_Fiction', 'Genre.Mystery', 
    'Genre.Music', 'Genre.Romance', 'Genre.Family', 'Genre.War', 'Genre.TV_Movie'
]

# Prepare features for KNN
features = df[genre_columns]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize KNN model
knn = NearestNeighbors(n_neighbors=10, metric='cosine')  # Adjust number of neighbors as needed
knn.fit(features_scaled)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_genre = None

    if request.method == 'POST':
        # Get selected genre from the dropdown
        selected_genre = request.form.get('genre')
        
        # Create a user profile with the selected genre
        user_genre_preferences = {genre: 1 if genre == selected_genre else 0 for genre in genre_columns}
        user_profile = pd.DataFrame([user_genre_preferences], columns=genre_columns).fillna(0)
        
        # Scale user profile
        user_profile_scaled = scaler.transform(user_profile)
        
        # Find nearest neighbors
        distances, indices = knn.kneighbors(user_profile_scaled)
        
        # Get recommended movie titles and poster paths
        recommended_movies = df.iloc[indices[0]][['title', 'poster_path']]
        
        # Build the list of recommendations with title and poster URL
        recommendations = [
            {
                'title': row['title'],
                'poster_url': f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
            }
            for _, row in recommended_movies.iterrows()
        ]

    return render_template('index.html', recommendations=recommendations, genres=genre_columns)

if __name__ == "__main__":
    app.run(debug=True)