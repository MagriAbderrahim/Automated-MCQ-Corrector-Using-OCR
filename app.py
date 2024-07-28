from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import tempfile
import pytesseract
from skimage import io
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import mysql.connector


app = Flask(__name__)

# Configuration de la base de données MySQL




UPLOAD_FOLDER = 'static/uploads/qcm'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('model.h5')
dict_word = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


def get_mysql_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            port='3306',
            user='root',
            password='',
            database='qcm'
        )
        print("Connexion à MySQL réussie !")
        return conn
    except mysql.connector.Error as e:
        print(f"Erreur de connexion à MySQL : {e}")
        return None

def recognize_text_from_image(image_path, model, dict_word, margin=5):
    """
    Recognizes text in an image by segmenting characters and predicting each character.

    Args:
        image_path (str): The path to the image.
        model (keras.Model): The pre-trained model for character recognition.
        dict_word (dict): The dictionary mapping model output to characters.
        margin (int): The margin to add around each character bounding box.

    Returns:
        str: The recognized text.
    """
    def recognize_character(character_image):
        # Resize the image to a standard size.
        character_image = cv2.resize(character_image, (28, 28))

        # Normalize the pixel values.
        character_image = character_image.astype('float32') / 255.0

        # Reshape the image to the model input format.
        character_image = np.expand_dims(character_image, axis=0)
        character_image = np.expand_dims(character_image, axis=-1)

        # Predict the character using the model.
        prediction = model.predict(character_image)

        # Get the predicted character.
        predicted_character = dict_word[np.argmax(prediction)]
        

        return predicted_character

    # Load image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Use morphology to enhance characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Sort contours from left to right
    cntrs = sorted(cntrs, key=lambda c: cv2.boundingRect(c)[0])

    # Extract individual characters and recognize them
    recognized_text = ""
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        x -= margin  # add or subtract margin to make the bounding box larger or smaller
        y -= margin
        w += 2 * margin
        h += 2 * margin
        char_img = thresh[max(0, y):min(y + h, thresh.shape[0]), max(0, x):min(x + w, thresh.shape[1])]
        recognized_char = recognize_character(char_img)
        recognized_text += recognized_char
 

    return recognized_text

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify(success=False, message="No file part")

    file = request.files['image']
    if file.filename == '':
        return jsonify(success=False, message="No selected file")

    if file:
       
        # Generate a unique filename using a counter or timestamp
        existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
        file_count = len(existing_files) + 1
        filename = f"qcm_{file_count}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify(success=True, filename=filename)

@app.route('/static/uploads/qcm/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_latest_qcm_file', methods=['GET'])
def get_latest_qcm_file():
    # Get a list of all files in the UPLOAD_FOLDER
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    # Filter out only qcm files and sort them
    qcm_files = [f for f in files if f.startswith('qcm_')]
    qcm_files.sort()
    
    # Get the latest qcm file
    if qcm_files:
        latest_file = qcm_files[-1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], latest_file)
        return jsonify({'filepath': filepath})
    else:
        return jsonify({'error': 'Aucun fichier QCM trouvé.'})

@app.route('/process_qcm', methods=['POST'])
def process_qcm():
    data = request.get_json()
    image_path = data.get('image_path', '')
    if not image_path:
        return jsonify({'success': False, 'message': 'Aucun chemin d\'image fourni.'})

    try:
        image = Image.open(image_path)
    except Exception as e:
        return jsonify({'success': False, 'message': f"Erreur: Impossible de lire l'image à partir de {image_path}. Détails: {str(e)}"})

    roi_dict = {
        'nom': (474, 204, 509, 64),
        'prenom': (474, 313, 507, 60),
        'reponses': [
            (382, 789, 198, 82), (382, 889, 198, 82), (382, 981, 198, 82), (382, 1075, 198, 82),
            (382, 1166, 198, 82), (382, 1263, 198, 82), (382, 1359, 198, 82), (385, 1455, 198, 82),
            (383, 1548, 198, 82), (382, 1641, 198, 82), (892, 783, 200, 78), (892, 877, 200, 78),
            (892, 976, 200, 78), (892, 1070, 200, 78), (892, 1162, 200, 78), (892, 1257, 200, 78),
            (892, 1355, 200, 78), (892, 1449, 200, 78), (892, 1542, 200, 78), (892, 1635, 200, 78)
        ]
    }

    infos_etudiant = []
    reponses_etudiant = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for key, value in roi_dict.items():
            if key in ['nom', 'prenom']:
                x, y, w, h = value
                roi = image.crop((x, y, x+w, y+h))
                temp_path = os.path.join(temp_dir, f"{key}.png")
                roi.save(temp_path)
                text = recognize_text_from_image(temp_path, model, dict_word)  # À remplacer par votre fonction de reconnaissance de texte
                infos_etudiant.append(f"{key.capitalize()}: {text}")
            elif key == 'reponses':
                for idx, coords in enumerate(value):
                    x, y, w, h = coords
                    roi = image.crop((x, y, x+w, y+h))
                    temp_path = os.path.join(temp_dir, f"response_{idx}.png")
                    roi.save(temp_path)
                    text = recognize_text_from_image(temp_path, model, dict_word)  # À remplacer par votre fonction de reconnaissance de texte
                    reponses_etudiant.append(text)

    # Établir une connexion à MySQL
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)

        # Récupérer les réponses correctes depuis la base de données
        cursor.execute("SELECT * FROM reponses_correct ORDER BY id DESC LIMIT 1")
        reponses_correct = cursor.fetchone()
        if not reponses_correct:
            return jsonify({'success': False, 'message': 'Aucune réponse correcte trouvée.'}), 500

        # Convertir les réponses correctes en liste
        reponses = [reponses_correct[f'reponse{i+1}'] for i in range(20)]

        # Calculer le score
        score = sum(1 for student, correct in zip(reponses_etudiant, reponses) if student == correct)

        # Extraire le nom et le prénom de l'étudiant
        nom = infos_etudiant[0].split(': ')[1]
        prenom = infos_etudiant[1].split(': ')[1]

        # Insérer les données de l'étudiant dans la table etudiant
        insert_query = "INSERT INTO etudiant (nom, prenom, score) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (nom, prenom, score))
        conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'success': False, 'message': f"Erreur lors de l'interaction avec la base de données: {str(e)}"}), 500

    return jsonify({'success': True, 'infos_etudiant': infos_etudiant, 'reponses_etudiant': reponses_etudiant, 'score': score})


@app.route('/save_correct_answers', methods=['POST'])
def save_correct_answers():
    data = request.get_json()
    reponses = data.get('reponses', [])
    print(reponses)

    # Validez si toutes les réponses sont remplies
    if all(reponses):
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()

            # Préparez la requête SQL avec les paramètres de la liste `reponses`
            query = """INSERT INTO reponses_correct (
                        reponse1, reponse2, reponse3, reponse4, reponse5,
                        reponse6, reponse7, reponse8, reponse9, reponse10,
                        reponse11, reponse12, reponse13, reponse14, reponse15,
                        reponse16, reponse17, reponse18, reponse19, reponse20)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            
            # Exécutez la requête SQL avec les valeurs de `reponses`
            cursor.execute(query, reponses)
            conn.commit()

            cursor.close()
            conn.close()

            return jsonify({'message': 'Réponses correctes enregistrées avec succès!'})
        except Exception as e:
            print(f"Erreur lors de l'insertion dans la base de données: {e}")
            return jsonify({'error': 'Erreur lors de l\'enregistrement des réponses correctes.'}), 500
    else:
        return jsonify({'error': 'Veuillez remplir tous les champs de réponse.'}), 400
@app.route('/viderTableEtudiants', methods=['POST'])
def viderTableEtudiants():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Supprimer toutes les entrées de la table etudiant
        delete_query = "DELETE FROM etudiant"
        cursor.execute(delete_query)
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'message': 'Table etudiant vidée avec succès!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f"Erreur lors de la suppression des données de la table etudiant: {str(e)}"}), 500
@app.route('/gets_students', methods=['GET'])
def get_students():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM etudiant")
        students = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(students)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
if __name__ == '__main__':
    app.run(debug=True)
