"""
explainer.py
------------

SigLearn AI Engine

This module manages:
    • Current biosignal analysis
    • Conversation memory
    • Context generation
    • Streaming chat
    • Educational summaries

The UI (Streamlit) never talks directly to the LLM.
It only interacts with the AIChat class.
"""

import os

from dotenv import load_dotenv
from copy import deepcopy

from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """
You are SigLearn, an expert biomedical signal tutor.

Your role is to teach biomedical engineering students using the
uploaded biosignal and the extracted analysis.

IMPORTANT RULES

1. Never diagnose patients.
2. Never invent measurements.
3. If information is unavailable, clearly state it.
4. Always explain physiology.
5. Always explain clinical relevance.
6. Relate every explanation to the CURRENT uploaded signal.
7. Encourage learning instead of simply answering.
8. If appropriate, suggest a follow-up concept the student should explore.

Your tone should be:
- Friendly
- Professional
- Educational
- Curious
"""


class AIChat:

    def __init__(self):

        self.model = "Qwen/Qwen2.5-7B-Instruct"

        print("HF_TOKEN =", os.getenv("HF_TOKEN"))

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
        )

        self.system_prompt = SYSTEM_PROMPT

        self.temperature = 0.4

        self.analysis = None

        self.analysis_loaded = False

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    @property
    def has_analysis(self):
        return self.analysis is not None
    
    # ==========================================================
    # ANALYSIS MANAGEMENT
    # ==========================================================

    def load_analysis(
        self,
        signal_type,
        classification,
        parameters,
        processing_steps,
        annotations,
        metadata=None
    ):
        """
        Store the current biosignal analysis.

        No LLM call happens here.
        """

        self.analysis = {
            "signal_type": signal_type,
            "classification": classification,
            "parameters": parameters,
            "processing_steps": processing_steps,
            "annotations": annotations,
            "metadata": metadata or {}
        }

        self.analysis_loaded = True

        # Every new signal starts a fresh conversation.
        self.reset()


    # ==========================================================
    # CONTEXT BUILDER
    # ==========================================================

    def build_context(self):
        """
        Build a detailed context describing the current analysis.

        This context is automatically supplied to the LLM so that
        every answer is specific to THIS uploaded signal.
        """

        if not self.analysis_loaded:
            return "No signal has been analysed yet."

        analysis = self.analysis

        signal_type = analysis["signal_type"]
        classification = analysis["classification"]
        parameters = analysis["parameters"]
        steps = analysis["processing_steps"]
        annotations = analysis["annotations"]
        metadata = analysis["metadata"]

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        confidence = classification.get("confidence", 0.0) * 100

        probability_text = "\n".join(
            f"- {label}: {prob * 100:.1f}%"
            for label, prob in classification.get("all_probs", {}).items()
        )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        if parameters:

            parameter_text = "\n".join(
                (
                    f"- {p['name']}\n"
                    f"    Value : {p['value']} {p['unit']}\n"
                    f"    Normal: {p['normal']}"
                )
                for p in parameters
            )

        else:

            parameter_text = "No extracted parameters."

        # --------------------------------------------------
        # Processing Steps
        # --------------------------------------------------

        if steps:

            processing_text = "\n".join(
                f"{i+1}. {step}"
                for i, step in enumerate(steps)
            )

        else:

            processing_text = "No processing steps available."

        # --------------------------------------------------
        # Detected Events
        # --------------------------------------------------

        if annotations:

            annotation_text = "\n".join(
                f"- {a['name']} ({a['time']:.2f} s)"
                for a in annotations
            )

        else:

            annotation_text = "No annotated events detected."

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        if metadata:

            metadata_text = "\n".join(
                f"- {k}: {v}"
                for k, v in metadata.items()
            )

        else:

            metadata_text = "No metadata available."

        # --------------------------------------------------
        # Final Context
        # --------------------------------------------------

        context = f"""
        CURRENT BIOSIGNAL ANALYSIS

        ==================================================

        Signal Type
        -----------
        {signal_type}

        Classification
        --------------
        Predicted Label : {classification.get("label")}
        Confidence      : {confidence:.1f} %

        Class Probabilities
        -------------------
        {probability_text}

        ==================================================

        Extracted Parameters
        --------------------
        {parameter_text}

        ==================================================

        Processing Pipeline
        -------------------
        {processing_text}

        ==================================================

        Detected Events
        ---------------
        {annotation_text}

        ==================================================

        Signal Metadata
        ---------------
        {metadata_text}

        ==================================================

        Instructions

        Use ONLY the information above while answering.

        Always relate explanations to THIS uploaded signal.

        If the student asks a conceptual question,
        connect the concept back to the current signal
        whenever possible.

        Never invent parameter values.

        Never diagnose a patient.

        This application is for education.
        """

        return context
    
        # ==========================================================
    # CHAT
    # ==========================================================

    def ask(
        self,
        question: str,
        temperature: float | None = None,
        max_tokens: int = 600,
    ):
        """
        Ask the AI a question.

        The conversation remembers previous messages.
        The current signal analysis is automatically injected.
        """

        if temperature is None:
            temperature = self.temperature

        # Inject analysis context once per conversation
        if self.analysis_loaded and len(self.messages) == 1:

            self.messages.append(
                {
                    "role": "system",
                    "content": self.build_context()
                }
            )

        # Add user message
        self.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        try:

            response = self.client.chat.completions.create(

                model=self.model,

                messages=self.messages,

                temperature=temperature,

                max_tokens=max_tokens,

            )

            reply = response.choices[0].message.content.strip()

            self.messages.append(
                {
                    "role": "assistant",
                    "content": reply
                }
            )

            return reply

        except Exception as e:

            # Remove failed user message
            self.messages.pop()

            return f"⚠️ AI unavailable:\n\n{e}"


    # ==========================================================
    # STREAMING CHAT
    # ==========================================================

    def ask_stream(
        self,
        question: str,
        temperature: float | None = None,
        max_tokens: int = 600,
    ):
        """
        Streaming version of ask().

        Yields text chunks one by one.
        """

        if temperature is None:
            temperature = self.temperature

        if self.analysis_loaded and len(self.messages) == 1:

            self.messages.append(
                {
                    "role": "system",
                    "content": self.build_context()
                }
            )

        self.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        full_reply = ""

        try:

            stream = self.client.chat.completions.create(

                model=self.model,

                messages=self.messages,

                temperature=temperature,

                max_tokens=max_tokens,

                stream=True,

            )

            for chunk in stream:

                delta = chunk.choices[0].delta.content

                if delta:

                    full_reply += delta

                    yield delta

            self.messages.append(
                {
                    "role": "assistant",
                    "content": full_reply
                }
            )

        except Exception as e:

            self.messages.pop()

            yield f"⚠️ AI unavailable:\n\n{e}"

        # ==========================================================
    # CONVERSATION MANAGEMENT
    # ==========================================================

    def reset(self):
        """
        Reset ONLY the conversation.

        The current signal analysis remains loaded.

        Example
        -------
        EEG loaded
            ↓
        Student asks 20 questions
            ↓
        reset()
            ↓
        Conversation starts fresh
            ↓
        Analysis is still available
        """

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]


    # ==========================================================
    # ANALYSIS MANAGEMENT
    # ==========================================================

    def clear_analysis(self):
        """
        Completely clear the loaded signal analysis
        and reset the conversation.
        """

        self.analysis = None
        self.analysis_loaded = False

        self.reset()


    # ==========================================================
    # CHAT EXPORT
    # ==========================================================

    def export_chat(self):
        """
        Return a copy of the conversation.

        Useful later for:
            • Download Chat
            • Generate Report
            • Save Session
        """

        return deepcopy(self.messages)


    # ==========================================================
    # INITIAL AI SUMMARY
    # ==========================================================

    def generate_initial_summary(self):
        """
        Generate an educational overview of the
        CURRENT analysed signal.

        This is automatically called after
        'Load & Analyze' in the Streamlit UI.

        Implemented fully in Step 5.
        """

        if not self.analysis_loaded:

            return "No signal has been analysed."

        prompt = """
        You are looking at a freshly analysed biosignal.

        Write an educational summary.

        Structure your response like this:

        1. Overall summary

        2. Explain the predicted signal type.

        3. Explain the confidence.

        4. Explain the important parameters.

        5. Mention interesting detected events.

        6. Explain why the preprocessing steps were used.

        Keep the explanation educational.

        Avoid diagnosis.
        """

        return self.ask(prompt)