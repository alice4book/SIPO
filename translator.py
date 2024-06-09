from googletrans import Translator


class TranslatorModule:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text: str, src: str = 'en', dest: str = 'pl') -> str:
        """
        Translates text from source language to destination language, handling texts longer than 5000 characters.

        :param text: The text to translate.
        :param src: Source language (default is English).
        :param dest: Destination language (default is Polish).
        :return: Translated text.
        """
        max_length = 5000
        if len(text) <= max_length:
            return self.translate_chunk(text, src, dest)

        # Split text into chunks of max_length characters
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        translated_chunks = [self.translate_chunk(chunk, src, dest) for chunk in chunks]

        # Combine all translated chunks
        translated_text = ''.join(translated_chunks)
        return translated_text

    def translate_chunk(self, text: str, src: str = 'en', dest: str = 'pl') -> str:
        """
        Translates a chunk of text from source language to destination language.

        :param text: The text chunk to translate.
        :param src: Source language (default is English).
        :param dest: Destination language (default is Polish).
        :return: Translated text.
        """
        translation = self.translator.translate(text, src=src, dest=dest)
        return translation.text
