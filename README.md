## Student name: Prince Gurung
## Student number: 2317057
## Project title: Smart Crop Recommendation System
## Link to project video recording: https://drive.google.com/file/d/13LcVVdfAACTudYVQeRZi5TeGJzupJHkE/view?usp=sharing

> **A Student-Friendly AI Project**  
> Predicts the best crop to cultivate based on soil and weather data using Machine Learning.

---

## 📌 Project Overview
This project is an application of **Machine Learning** in Agriculture. It helps farmers decide which crop will give the best yield by ensuring their soil and climate parameters (like Nitrogen, pH, Rainfall) match the crop's requirements.

**Techniques Used:**
- **Machine Learning**: Random Forest Classifier
- **Web Development**: Flask (Python)
- **Data Processing**: Pandas & Scikit-learn

For a detailed explanation of the **algorithms and concepts**, please read [docs.md](./docs.md).

---

## 📂 Directory Structure

```
Crop-Recommendation/
├── app.py                # The main Flask application (runs the website)
├── train_model.py        # The script to train the Machine Learning model
├── Crop_recommendation.csv # The dataset used for training
├── model.pkl             # The trained model (frozen brain)
├── minmaxscaler.pkl      # The scaler (helper for numbers)
├── requirements.txt      # List of libraries required
├── docs.md               # Detailed student documentation
├── templates/            # HTML files (Index, Result pages)
└── static/               # CSS, Images
```

---

## 🛠️ How to Run This Project

### Step 1: Install Python
Make sure you have Python installed on your computer. You can check by running:
```bash
python --version
```

### Step 2: Install Libraries
Open your terminal (or command prompt) in this folder and run:
```bash
pip install -r requirements.txt
```
*This installs all the necessary tools like Flask, Pandas, and Scikit-learn.*

### Step 3: Run the Application
Start the website by running:
```bash
python app.py
```
You should see output like:
`Running on http://127.0.0.1:5000`

### Step 4: Use the App
Open your browser (Chrome/Edge) and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

Enter values (or random test values) and hit **Predict**!

---

## 🧪 How to Retrain the Model (Optional)
If you want to experiment or change the algorithm, you can re-train the model.
1. Open `train_model.py`.
2. Modified the code (e.g., change `RandomForestClassifier` to `DecisionTreeClassifier`).
3. Run the training script:
   ```bash
   python train_model.py
   ```
   *This will update `model.pkl` with your new logic.*

---

## 📝 Credits
- **Dataset**: Crop Recommendation Dataset
- **Developed for**: Educational Purposes

Happy Coding! 🚀
