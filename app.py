from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from forms import LoginForm, SignupForm
from process import save_post
from datetime import datetime
from video_process import get_video_prediction

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///tweets.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    content = db.Column(db.String(280), nullable=False)
    image = db.Column(db.String(120), nullable=True)
    video = db.Column(db.String(120), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize the database
with app.app_context():
    db.create_all()
    print("Database initialized")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user is None:
            hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
            new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for("login"))
        else:
            flash("An account with this email already exists.", "danger")
    return render_template("signup.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("index"))
        else:
            flash("Login failed. Check your email and password.", "danger")
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        return redirect(url_for("tweet"))
    tweets = Tweet.query.order_by(Tweet.timestamp.desc()).all()
    return render_template("index.html", tweets=tweets)

@app.route("/tweet", methods=["POST"])
@login_required
def tweet():
    content = request.form.get("tweetText")
    image = request.files.get("imageUpload")
    video = request.files.get("videoUpload")

    try:
        new_tweet_data = save_post(content, image, video)
        # print(new_tweet_data)
        new_tweet = Tweet(
            username=current_user.username,
            content=new_tweet_data["content"],
            image=new_tweet_data["image"],
            video=new_tweet_data["video"],
            timestamp=new_tweet_data["timestamp"]
        )
        uploads_folder='static\\uploads'
        print("#############################################")
        video_path=os.path.join(uploads_folder, new_tweet_data["video"])
        print(video_path)
        print("################################################")
        get_video_prediction(video_path=video_path)
        db.session.add(new_tweet)
        db.session.commit()

    except ValueError as e:
        return str(e), 400

    return redirect(url_for("index"))

@app.route("/api/tweets", methods=["GET"])
def get_tweets():
    tweets = Tweet.query.order_by(Tweet.timestamp.desc()).all()
    tweet_list = [
        {
            "id": tweet.id,
            "username": tweet.username,
            "content": tweet.content,
            "image": tweet.image,
            "video": tweet.video,
            "timestamp": tweet.timestamp
        }
        for tweet in tweets
    ]
    return jsonify(tweet_list)

@app.route("/api/tweet", methods=["POST"])
def create_tweet():
    data = request.json

    if "content" not in data:
        return jsonify({"error": "Content is required"}), 400

    new_tweet = Tweet(
        username=data.get("username", "DefaultUser"),
        content=data["content"],
        image=data.get("image"),
        video=data.get("video")
    )

    db.session.add(new_tweet)
    db.session.commit()

    return jsonify({"message": "Tweet created successfully"}), 201

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)
