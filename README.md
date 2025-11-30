🌌 CineVerse AI
Hybrid AI-Powered Movie Recommendation System with 3D Futuristic UI

CineVerse AI is a full-stack movie recommendation platform that blends Machine Learning, Flask backend, MySQL database, and a stunning 3D dark-mode UI. Built with hybrid recommendation algorithms, CineVerse AI provides intelligent movie suggestions based on user preferences, ratings, similar movie patterns, and genre interests — just like Netflix and Prime Video.

🚀 Features
🎯 Recommendation Engine

✔ Content-Based Filtering using TF-IDF + Cosine Similarity
✔ Collaborative Filtering (User-Item patterns)
✔ NMF Latent Factorization for hidden movie insights
✔ Autoencoder-based Deep Learning for reconstruction-driven recommendations
✔ Hybrid Scoring combining ML models for best accuracy

🖥️ Frontend (Lovable UI + Custom CSS)

✔ 3D animated CineVerse AI title
✔ Dark-themed futuristic UI
✔ Movie posters, genre badges, search bar

⚙️ Backend (Flask)

✔ REST APIs for authentication, movies, ratings, recommendations
✔ Session-based login system
✔ Watchlist toggle and user ratings

🗄️ Database (MySQL)

✔ Users table
✔ Movies table
✔ Ratings table
✔ Watchlist table
✔ External image posters stored via URL

🧱 Tech Stack
Layer	Technologies
Frontend	HTML, CSS, Lovable AI UI
Backend	Flask (Python)
ML Models	NMF, Autoencoder, TF-IDF, Cosine Similarity
Database	MySQL
Deployment	Local / Future Cloud Support
📂 Project Structure
CineVerse-AI/
│
├── backend/
│   ├── app.py
│   ├── db.py
│   ├── models/        # Place downloaded ML models here
│   └── templates/     # HTML pages
│
├── static/
│   └── css/style.css  # 3D UI styling
│
├── requirements.txt
└── README.md

📦 ML Model Download (Required)

Trained ML models are stored externally (GitHub limit exceeded).

🔗 Download here:
https://drive.google.com/drive/folders/1tIL9aXB9JKHq4yo4uPC5Op6VuXFNGHLn?usp=sharing

After download, extract and place files into:

backend/models/


Your directory must contain:

autoencoder_cf.h5
content_similarity.pkl
nmf_movie_factors.npy
movie_list.pkl
🔐 User Features
Feature	Description
Signup / Login	Authentication via phone + password
Home Page	Top rated movies + CineVerse UI
Search	Find movies by name
Genre Filter	Discover by Action, Sci-Fi, Romance, etc.
Movie Page	Overview, ratings, add/remove watchlist
Personalized Recommendations	Based on your ratings
Watchlist	Your saved movies
🧠 Hybrid Recommendation Logic
final_score = 0.5 * content_based
             + 0.3 * NMF_latent_features
             + 0.2 * Autoencoder_predictions


This solves:

✔ cold-start problem
✔ multi-user similarity issues
✔ personalized ranking

🎯 Future Enhancements

🔜 Mobile App (Flutter / PWA)
🔜 Social recommendations
🔜 Real-time retraining
🔜 Trending & regional suggestions
🔜 Voice-based movie search

🏆 Why This Project Is Valuable

✔ Production-grade ML integration
✔ Real-world recommendation pipeline
✔ End-to-end full-stack deployment
✔ Excellent resume + portfolio project
✔ Demonstrates ML + Backend + UI mastery

💡 Author

Harshitha S
CineVerse AI — Where Movies Meet Intelligence 🍿🤖

⭐ Support

If this project helped you, star the repo ⭐
Your support motivates future updates!
