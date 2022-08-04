import pyttsx3
import speech_recognition as sr

import train_data
import  util


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try :
        print("Recognizing...")
        query = r.recognize_google(audio, language = 'en-in')
        print(f"user said: {query}\n")

    except Exception as e:
        print("Say that again please...")
        query = None

    return query
def main():
    speak("Initializing radlid...")
    while True :
        query = takeCommand()
        while query is None:
            speak('oh sorry! i am confused,please repeat again!')
            query = takeCommand()
        print(query)
        resp = util.classify(query)
        msg = resp[0].get('intent')
        ans = train_data.findResponse(resp[0].get('intent'))
        print(*resp, sep='\n')
        print(msg)
        speak(ans)
if __name__ == '__main__':
    main()