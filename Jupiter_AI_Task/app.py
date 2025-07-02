from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Credentials
openai.api_key = 'open-ai-key'

@app.route('/', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']

    # Responses
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )

    return jsonify({'response': response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run()
