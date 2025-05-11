import re
import ollama
from src.config import EMBEDDING_MODEL, LANGUAGE_MODEL


class RAGSystem(object):
    # Clase que implementa un sistema RAG (Retrieval-Augmented Generation) sencillo.
    # Utiliza embeddings para almacenar y recuperar información relevante basada en similitud semántica.

    # Base de datos vectorial donde se almacenan los fragmentos de texto y sus embeddings correspondientes.
    VECTOR_DB = []

    def __init__(self, dataset_file):
        """
        Constructor de la clase RAGSystem.
        Carga un conjunto de datos desde un archivo y lo almacena en la base de datos vectorial.

        :param dataset_file: Ruta del archivo que contiene los datos a cargar.
        """
        self.dataset = []
        self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        """
        Carga el contenido de un archivo en la variable dataset y lo almacena en la base de datos vectorial.

        :param dataset_file: Ruta del archivo que contiene los datos a cargar.
        """
        with open(dataset_file, 'r', encoding='utf-8') as file:
            self.dataset = file.readlines()
            print(f'Loaded {len(self.dataset)} entries')

        index = 0
        for i, chunk in enumerate(self.dataset):
            if chunk.strip():
                normalized = self.normalize_text(chunk)
                self.add_chunk_to_database(normalized, index)
                print(f'Added chunk {index}: {normalized[:60]}...')
                index += 1

    def normalize_text(self, text):
        return re.sub(r'\s+', ' ', text.strip().lower())

    def add_chunk_to_database(self, chunk, index):
        """
        Convierte un fragmento de texto en un embedding y lo almacena en la base de datos vectorial.

        :param chunk: Fragmento de texto a agregar.
        """
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        self.VECTOR_DB.append({
            'index': index,
            'text': chunk,
            'embedding': embedding
        })

    def cosine_similarity(self, a, b):
        """
        Calcula la similitud coseno entre dos vectores.

        :param a: Primer vector.
        :param b: Segundo vector.
        :return: Valor de similitud coseno entre los vectores a y b.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    def retrieve_function(self, query, top_n=3):
        """
        Busca en la base de datos los fragmentos más relevantes en función de una consulta.

        :param query: Texto de la consulta.
        :param top_n: Número de fragmentos más relevantes a devolver.
        :return: Lista con los fragmentos más similares a la consulta.
        """
        # Obtener embedding de la consulta.
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

        # Lista para almacenar las similitudes entre la consulta y cada fragmento en la base de datos.
        similarities = []

        for entry in self.VECTOR_DB:
            similarity = self.cosine_similarity(query_embedding, entry['embedding'])
            similarities.append((entry['index'], entry['text'], similarity))

        # Ordenar los fragmentos por similitud en orden descendente (mayor similitud primero).
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_n]

    def ask_question(self, query, max_results_ranking):
        """
        Responde a una pregunta basada en los fragmentos más relevantes encontrados en la base de datos.

        :param query: Pregunta del usuario.
        :param max_results_ranking: Número de fragmentos relevantes a considerar en la respuesta.
        """
        retrieved_knowledge = self.retrieve_function(query, max_results_ranking)

        print('Retrieved knowledge:')
        for index, chunk, similarity in retrieved_knowledge:
            print(f' - [#{index}] (similarity: {similarity:.2f}) {chunk}')

        context = '\n'.join([f' - [#{index}] {chunk}' for index, chunk, _ in retrieved_knowledge])

        # Crear el prompt con la información recuperada
        instruction_prompt = f"""Eres un chatbot útil. Usa solo las siguientes partes del contexto para responder la pregunta.  
No inventes información nueva:  
{context}"""

        # Enviar la consulta al modelo de lenguaje con el contexto recuperado.
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )

        # Mostrar la respuesta del chatbot en tiempo real.
        print('Respuesta del Chatbot:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
