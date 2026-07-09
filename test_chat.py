from ai.explainer import AIChat

chat = AIChat()

chat.load_analysis(

    signal_type="EEG",

    classification={
        "label":"EEG",
        "confidence":0.42,
        "all_probs":{
            "EEG":0.42,
            "EMG":0.38
        }
    },

    parameters=[
        {
            "name":"Alpha Power",
            "value":15.2,
            "unit":"%",
            "normal":"Relaxed wakefulness"
        }
    ],

    processing_steps=[
        "Bandpass",
        "Notch",
        "Normalize"
    ],

    annotations=[
        {
            "name":"Alpha Burst",
            "time":2.41
        }
    ]

)

print(chat.ask(

    "Explain this signal."

))

print(

    chat.ask(

        "Why is Alpha Power important?"

    )

)

print(

    chat.ask(

        "How does it compare with Beta?"

    )

)