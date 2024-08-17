from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

app = Flask(__name__)

# Rebuild the CNN model (example architecture)
model_cnn = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(14, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

# Assuming model is already trained and loaded


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form data
        first_term_gpa = float(request.form['first_term_gpa'])
        second_term_gpa = float(request.form['second_term_gpa'])
        first_language = int(request.form['first_language'])
        funding = int(request.form['funding'])
        school = int(request.form['school'])
        fasttrack = int(request.form['fasttrack'])
        coop = int(request.form['coop'])
        residency = int(request.form['residency'])
        gender = int(request.form['gender'])
        previous_education = int(request.form['previous_education'])
        age_group = int(request.form['age_group'])
        high_school_average_mark = float(
            request.form['high_school_average_mark'])
        math_score = float(request.form['math_score'])
        english_grade = float(request.form['english_grade'])

        # Create a numpy array from the input and reshape it for the CNN model
        input_data = np.array([[first_term_gpa, second_term_gpa, first_language, funding, school, fasttrack,
                                coop, residency, gender, previous_education, age_group, high_school_average_mark,
                                math_score, english_grade]])

        # Reshape the data for the Conv1D layer: (batch_size, time_steps, features)
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], 1))

        # Predict the target
        prediction = model_cnn.predict(input_data)
        predicted_class = (prediction > 0.5).astype(int)

        return render_template('result.html', prediction=prediction[0][0], predicted_class=predicted_class[0][0])

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
