from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import numpy as np
import pickle
import os

from db import get_db_connection
from keras.models import load_model   # using keras load_model directly


app = Flask(__name__)
app.secret_key = "5224e335f617c7b1957a1096a0c7b2b21e2c5f994f917c8485bbbe5fd241945e"


# ============================ MODEL LOADING ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # backend/
MODEL_DIR = os.path.join(BASE_DIR, "models")              # backend/models/

# Load similarity model
content_sim = pickle.load(open(os.path.join(MODEL_DIR, "content_similarity.pkl"), "rb"))

# Load movie index mapping
movie_list = pickle.load(open(os.path.join(MODEL_DIR, "movie_list.pkl"), "rb"))

# Load NMF movie latent vectors
nmf_movie_factors = np.load(os.path.join(MODEL_DIR, "nmf_movie_factors.npy"))

# Load autoencoder WITHOUT compiling (avoids mse error)
autoencoder = load_model(
    os.path.join(MODEL_DIR, "autoencoder_cf.h5"),
    compile=False
)

# Map index → movie_id
if isinstance(movie_list, dict):
    index_to_movie_id = {v: k for k, v in movie_list.items()}
else:
    index_to_movie_id = {i: mid for i, mid in enumerate(movie_list)}

NUM_MOVIES = len(index_to_movie_id)


# ============================ HELPERS ============================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def normalize(arr):
    arr = np.array(arr, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ============================ AUTH ROUTES ============================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        password = request.form["password"]

        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # Check if phone already exists
        cur.execute("SELECT id FROM users WHERE phone=%s", (phone,))
        user = cur.fetchone()

        if user:
            return render_template("signup.html", error="Phone number already registered!")

        # Insert new user
        cur.execute(
            "INSERT INTO users (name, phone, password_hash) VALUES (%s, %s, %s)",
            (name, phone, generate_password_hash(password))
        )
        conn.commit()

        # Fetch user id and store session
        cur.execute("SELECT id FROM users WHERE phone=%s", (phone,))
        new_user = cur.fetchone()

        session["user_id"] = new_user["id"]
        session["user_name"] = name

        return redirect('/')

    return render_template("signup.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        phone = request.form["phone"]
        password = request.form["password"]

        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE phone=%s", (phone,))
        user = cur.fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect('/')

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect('/login')


# ============================ HOME PAGE ============================

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, title, poster_path, imdb_rating
        FROM movies
        ORDER BY imdb_rating DESC
        LIMIT 20
    """)
    movies = cur.fetchall()

    return render_template("home.html", movies=movies, user_name=session["user_name"])


# ============================ SEARCH & GENRE FILTER ============================

@app.route('/search')
@login_required
def search_movies():
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify([])

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT id, title, poster_path, imdb_rating
        FROM movies
        WHERE title LIKE %s
        LIMIT 15
    """, (f"%{query}%",))

    return jsonify(cur.fetchall())


@app.route('/genre/<genre>')
@login_required
def filter_genre(genre):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT id, title, poster_path, imdb_rating
        FROM movies
        WHERE genres LIKE %s
        ORDER BY imdb_rating DESC
        LIMIT 40
    """, (f"%{genre}%",))

    return jsonify(cur.fetchall())


# ============================ WATCHLIST (LIST) ============================

@app.route('/watchlist')
@login_required
def get_watchlist():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT m.id, m.title, m.poster_path, m.imdb_rating
        FROM watchlist w
        JOIN movies m ON m.id = w.movie_id
        WHERE w.user_id = %s
    """, (session['user_id'],))

    return jsonify(cur.fetchall())


# ============================ MOVIE PAGE ============================

@app.route('/movie/<int:id>')
@login_required
def movie_page(id):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM movies WHERE id=%s", (id,))
    movie = cur.fetchone()

    return render_template("movie.html", movie=movie, user_name=session["user_name"])


# ============================ PROFILE PAGE (NEW) ============================

@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    # User ratings joined with movies
    cur.execute("""
        SELECT m.id, m.title, m.poster_path, r.rating
        FROM ratings r
        JOIN movies m ON m.id = r.movie_id
        WHERE r.user_id = %s
        ORDER BY r.rating DESC
    """, (session["user_id"],))
    rated = cur.fetchall()

    # Watchlist joined with movies
    cur.execute("""
        SELECT m.id, m.title, m.poster_path, m.imdb_rating
        FROM watchlist w
        JOIN movies m ON m.id = w.movie_id
        WHERE w.user_id = %s
    """, (session["user_id"],))
    watchlist = cur.fetchall()

    return render_template(
        "profile.html",
        user_name=session["user_name"],
        rated=rated,
        watchlist=watchlist
    )


# ============================ RATINGS ============================

@app.route('/rate', methods=['POST'])
@login_required
def rate_movie():
    data = request.json
    movie_id = data["movie_id"]
    rating = data["rating"]

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO ratings (user_id, movie_id, rating)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE rating=%s
    """, (session["user_id"], movie_id, rating, rating))

    conn.commit()
    return jsonify({"success": True})


@app.route('/user-rating/<int:movie_id>')
@login_required
def get_user_rating(movie_id):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT rating FROM ratings WHERE user_id=%s AND movie_id=%s",
                (session["user_id"], movie_id))
    r = cur.fetchone()
    return jsonify({"rating": r["rating"] if r else None})


# ============================ WATCHLIST TOGGLE/STATUS ============================

@app.route('/watchlist/status/<int:movie_id>')
@login_required
def watchlist_status(movie_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM watchlist WHERE user_id=%s AND movie_id=%s",
                (session["user_id"], movie_id))
    return jsonify({"in_watchlist": bool(cur.fetchone())})


@app.route('/watchlist/toggle', methods=['POST'])
@login_required
def watchlist_toggle():
    data = request.json
    movie_id = data["movie_id"]

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM watchlist WHERE user_id=%s AND movie_id=%s",
                (session["user_id"], movie_id))
    row = cur.fetchone()

    if row:
        cur.execute("DELETE FROM watchlist WHERE id=%s", (row[0],))
        conn.commit()
        return jsonify({"in_watchlist": False})

    cur.execute(
        "INSERT INTO watchlist (user_id, movie_id) VALUES (%s, %s)",
        (session["user_id"], movie_id)
    )
    conn.commit()
    return jsonify({"in_watchlist": True})


# ============================ CONTENT SIMILAR MOVIES ============================

@app.route('/recommend/<int:movie_id>')
@login_required
def recommend(movie_id):
    idx = movie_list[movie_id] if isinstance(movie_list, dict) else movie_list.index(movie_id)
    sims = sorted(list(enumerate(content_sim[idx])), key=lambda x: x[1], reverse=True)[1:8]
    ids = [index_to_movie_id[i] for i, _ in sims]

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    q = ','.join(['%s'] * len(ids))
    cur.execute(f"SELECT id, title, poster_path FROM movies WHERE id IN ({q})", tuple(ids))
    return jsonify(cur.fetchall())


# ============================ HYBRID RECOMMENDER ============================

@app.route('/recommend/user')
@login_required
def recommend_user():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT movie_id, rating FROM ratings WHERE user_id=%s", (session["user_id"],))
    rated = cur.fetchall()

    if not rated:
        return jsonify([])

    user_vec = np.zeros(NUM_MOVIES)
    rated_idx = []

    for r in rated:
        mid, rat = r["movie_id"], float(r["rating"])
        idx = movie_list[mid] if isinstance(movie_list, dict) else movie_list.index(mid)
        user_vec[idx] = rat
        rated_idx.append(idx)

    # Content-based
    content_scores = sum(content_sim[i] * user_vec[i] for i in rated_idx)

    # NMF
    latent_user = np.dot(user_vec, nmf_movie_factors) / max(user_vec.sum(), 1e-6)
    nmf_scores = nmf_movie_factors @ latent_user

    # Autoencoder
    ae_scores = autoencoder.predict((user_vec / 5).reshape(1, -1), verbose=0)[0]

    # Combine
    final = 0.5 * normalize(content_scores) + 0.3 * normalize(nmf_scores) + 0.2 * normalize(ae_scores)
    final[rated_idx] = -1e9

    top = np.argsort(final)[::-1][:12]
    movies = [index_to_movie_id[i] for i in top]

    q = ",".join(["%s"] * len(movies))
    cur.execute("""
        SELECT id, title, poster_path, imdb_rating
        FROM movies WHERE id IN ({})
    """.format(q), tuple(movies))

    rows = cur.fetchall()
    order = {mid: i for i, mid in enumerate(movies)}
    rows.sort(key=lambda x: order[x["id"]])

    return jsonify(rows)


# ============================ RUN SERVER ============================

if __name__ == "__main__":
    app.run(debug=True)
