from flask import Flask,render_template,request
from src.helper import init_rag
from src.prompt import system_prompt

app = Flask(__name__)

rag_chain,chat_history = init_rag('./data/medical-book.pdf',system_prompt)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get',methods=['GET','POST'])
def chat():
    user_msg = request.form['msg']
    result = rag_chain.invoke({'input':user_msg,"chat_history":chat_history.messages})
    chat_history.add_user_message(user_msg)
    chat_history.add_ai_message(result['answer'])
    return result['answer']


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)