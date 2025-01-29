import os
from flask import Flask, render_template_string, render_template, request, jsonify
from googlesearch import search
import datetime
import speech_recognition as sr
from pydub import AudioSegment
import io
from openai import OpenAI
import sys
import requests
# File paths




app = Flask(__name__)

# Path to the txt file
file_path = "scratch.txt"

future_file = "future.txt"
month_file = "monthly.txt"
mission_file = "mission.txt"
misc_file = "misc.txt"
past_file = "past.txt"
reminder_file = "reminders.txt"
temp_task_file = "temp_tasks.txt"
daily_file = "daily.txt"


def query_chatgpt(prompt, llm, model1="gpt-4", max_tokens=1500):
    """
    Queries the OpenAI ChatGPT model and returns the response.

    Args:
        prompt (str): The input prompt to send to the model.
        model (str): The GPT model to use (e.g., "gpt-4" or "gpt-3.5-turbo").
        max_tokens (int): Maximum number of tokens in the response.

    Returns:
        str: The response from the model.
    """

    gpt = 1;
    if(llm=="ChatGPT"):

        model1 = "gpt-4o-mini"
        api_key = ""

        client = OpenAI(api_key=api_key)  # Replace with your actual API key

        # Non-streaming:
        print("----- standard request -----")
        completion = client.chat.completions.create(
            model=model1,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return [1, completion.choices[0].message.content]

    elif (llm=="deep_seek"):
        # Replace with your actual API key and endpoint
        API_KEY = 'your_deepseek_api_key_here'
        API_URL = 'https://api.deepseek.com/v1/endpoint'  # Replace with the actual API endpoint

        # Headers for the request
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        # Example payload (modify according to the API requirements)
        payload = {
            'prompt': prompt,
            'max_tokens': 5000
        }

        # Send a POST request to the DeepSeek API
        response = requests.post(API_URL, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            print("Response from DeepSeek API:")
            return [1, data]
        else:
            print(f"Failed to get a response. Status code: {response.status_code}")
            return [0, data]

    elif (llm=="Awan"):
        # Replace with your actual API key and endpoint
        API_KEY = ''
        API_URL = 'https://api.awanllm.com/v1/endpoint'  # Replace with the actual API endpoint

        # Headers for the request
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        # Example payload (modify according to the API requirements)
        payload = {
            'prompt': prompt,
            'max_tokens': 5000
        }

        # Send a POST request to the DeepSeek API
        response = requests.post(API_URL, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            print("Response from DeepSeek API:")
            return [1, data]
        else:
            print(f"Failed to get a response. Status code: {response.status_code}")
            return [0, response.text]

# Function to update the text file with search results
def daily_update():

    daily_file = "daily.txt"
    # Perform a Google search for "messi"
    prompt = daily_prompt()

    output = query_chatgpt(prompt)
    if (output[0]):
        with open(daily_file, "w") as file:
            file.write(output[1] + "\n")

# Function to update the text file with search results
def monthly_update():

    # Perform a Google search for "messi"
    now = datetime.datetime.now()

    # Format the date into a text string
    formatted_date = now.strftime("%B")

    future_file = "future.txt"
    month_file = "monthly.txt"
    mission_file = "mission.txt"

    with open(future_file, 'r') as file:
        file_content =  file.read()
    with open(mission_file, 'r') as file:
        file_content = file_content + file.read()
    with open(month_file, 'r') as file:
        template = file.read()

    prompt = (
                "Can you make a task list for "+formatted_date+ " in html format. The tasks pending from last month in the correct format is given below\n" + template +
                "The pending tasks from last month should be transferred to this month and needs to be in the same format. The new entries if needed needs to be created based on my long term future goals and my values.\n" \
                "They are attched below\n" + file_content)

    output = query_chatgpt(prompt)
    if (output[0]):
        with open(month_file, "w") as file:
            file.write(output[1] + "\n")

def daily_prompt():
    now = datetime.datetime.now()

    # Format the date into a text string
    formatted_date = now.strftime("%A, %B %d, %Y")

    future_file = "future.txt"
    month_file = "monthly.txt"
    mission_file = "mission.txt"
    misc_file = "misc.txt"
    reminder_file = "reminders.txt"
    temp_task_file = "temp_tasks.txt"
    template_file = "template.txt"

    with open(temp_task_file, 'r') as file:
        file_content = file.read()
    with open(reminder_file, 'r') as file:
        file_content = file_content + file.read()
    with open(misc_file, 'r') as file:
        file_content = file_content + file.read()
    with open(month_file, 'r') as file:
        file_content = file_content + file.read()
    with open(future_file, 'r') as file:
        file_content = file_content + file.read()
    with open(mission_file, 'r') as file:
        file_content = file_content + file.read()
    with open(template_file, 'r') as file:
        template =  file.read()

    prompt = ("Today is " + formatted_date + ". Can make a schedule of the whole day and reminders (only if a reminder is set for today) of the day in html format. The template for this is given below\n"+template+
              "Plaese follow the template and do not add pefix or suffixes in the response. This needs to be created based on my immediate tasks, monthly goals, my long term future goals and my values and reminders.\n" \
            "They are attched below\n" + file_content)

    return prompt
# Schedule the update (you can schedule this in the system cron or use a scheduler library)


def handle_user_query(text, llm):
    prompt = "The following is the list of documents that keep track of a lot of information of the user.\n"\
            "Mission: The summary of main goals and values of the user.\n"\
            "Future: The list of primary goals that need to be completed in the long term.\n"\
            "Monthly: The list of tasks that needs to be done in this month.\n"\
            "Temp: The list of immediate task that needs to be worked on\n"\
            "Remind: The list of reminders for the user\n"\
            "Misc: The description of different priorities or preferences that need to be accounted for when creating the daily timetable for the user.\n"\
            "Daily: The timetable of today containing the list of tasks and its assigned time.\n\n"\
            "The user has a query. Based on the query, can you list the names of the documents that needs to be edited if necessary and also describe what are the changes that need to be made in a table format? Can you only output the table and nothing else?\n\n"\
            "Query: \"" + text

    output = query_chatgpt(prompt, llm)
    dailyFlag = 0

    if (output[0]):
        list = output[1].splitlines()[2:]

        for i in range(len(list)):
            line = list[0].split("|")
            file_name = ''.join(e for e in line[1] if e.isalnum())
            string = line[2]
            if("remind" in file_name.lower()):
                file_name = reminder_file
            elif("monthly" in file_name.lower()):
                file_name = month_file
            elif("future" in file_name.lower()):
                file_name = future_file
            elif("mission" in file_name.lower()):
                file_name = mission_file
            elif("misc" in file_name.lower()):
                file_name = misc_file
            elif("temp" in file_name.lower()):
                file_name = temp_task_file
            elif("daily" in file_name.lower()):
                file_name = daily_file
                dailyFlag = 1

            with open(file_name, 'r') as file:
                file_content = file.read()

            if(dailyFlag):
                prompt = daily_prompt()
                prompt = prompt + ". I also want to incorporate the suggestion given below.\nSuggestion:\n" + string
            else:
                prompt = "I want to make some changes to a document based on a query. The format and the structure of the document should not change. Only the contents of the file should be updated. Can you update the file? Please dont add and prefaces or endnotes in the output.\n"\
                        "File:\n" +file_content+ "Query:\n" + string

            output = query_chatgpt(prompt, llm)
            if (output[0]):
                with open(file_name, "w") as file:
                    # Write the text to the file
                    file.write(output[1])

            prompt = "Does the following query sounds like the user has completed a task? PLease start the answer wiht a yes or a no. If it is a task completed, can you explain the completed task in detain after the yes?\n" \
                     "Query:\n" + string

            output = query_chatgpt(prompt, llm)
            if (output[0]):

                if (output[1][0].lower() == "y"):
                    with open(past_file, "a") as file:
                        file.write(output[1] + "\n")

print(str(len(sys.argv))+"\n")
if(len(sys.argv)>1):
    if(int(sys.argv[1])==1):
        daily_update()
    elif (int(sys.argv[1]) == 2):
        monthly_update()

# HTML form for submitting text
HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Text Submission</title>
</head>
<body>
    <h1>Submit Audio to Text</h1>
    <form action="/" method="POST">
        <textarea name="text_input" rows="5" cols="40" placeholder="Enter your text here..."></textarea><br><br>
        <input type="submit" value="Submit">
    </form>
    <h2>Record Audio</h2>
    <button id="recordButton">Start Recording</button>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def handle_text():
    if request.method == 'POST':
        text_input = request.form['text_input']
        llm = request.form['text_input']
        if text_input.strip():
            handle_user_query(text_input, llm)
            with open(file_path, "a") as file:
                file.write(f"\n{text_input}\n")
    return render_template_string('index.html')


@app.route('/record_audio', methods=['POST'])
def record_audio():

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400


    audio_file = request.files['audio']
    llm = request.files['selected_option']

    # Convert audio to wav format using pydub (if necessary)
    audio_data = audio_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_data))

    # Save as a temporary file
    temp_filename = "temp_audio.wav"
    audio.export(temp_filename, format="wav")

    # Use speech recognition to convert audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_filename) as source:
        audio = recognizer.record(source)

    try:
        # Convert audio to text using Google's Speech-to-Text
        transcript = recognizer.recognize_google(audio)

        handle_user_query(transcript, llm)

        with open(file_path, "a") as file:
            file.write(f"\n{transcript}\n")

        # Return the recognized text to the frontend
        return jsonify({"transcript": transcript})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Could not request results from Google Speech Recognition service"}), 500


@app.route('/get_text/<int:file_id>', methods=['GET'])
def get_text(file_id):
    try:
        # Map file IDs to filenames
        filenames = {
            1: 'daily.txt',
            2: 'monthly.txt',
            3: 'mission.txt',
            4: 'reminders.txt',
            5: 'future.txt',
            6: 'misc.txt',
            7: 'temp_tasks.txt'

        }
        filename = filenames.get(file_id)
        if not filename:
            return jsonify({'error': 'Invalid file ID'}), 404

        # Read the content of the file
        with open(filename, 'r') as file:
            content = file.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)