import ollama


class RAGSystem(object):
    # Clase que implementa un sistema RAG (Retrieval-Augmented Generation) sencillo.
    # Utiliza embeddings para almacenar y recuperar información relevante basada en similitud semántica.

    # Modelos de vectores de embeddings y modelo de lenguaje
    EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'  # Modelo de embeddings utilizado para representar los textos como vectores.
    LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'  # Modelo de lenguaje utilizado para responder preguntas basadas en la información recuperada.

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
            self.dataset = file.readlines()  # Leer todas las líneas del archivo como una lista de strings.
            print(f'Loaded {len(self.dataset)} entries')

        # Procesar cada fragmento de texto y agregarlo a la base de datos.
        for i, chunk in enumerate(self.dataset):
            self.add_chunk_to_database(chunk)
            print(f'Added chunk {i + 1}/{len(self.dataset)} to the database')

    def add_chunk_to_database(self, chunk):
        """
        Convierte un fragmento de texto en un embedding y lo almacena en la base de datos vectorial.

        :param chunk: Fragmento de texto a agregar.
        """
        embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][
            0]  # Obtener el embedding del fragmento.
        self.VECTOR_DB.append((chunk, embedding))  # Guardar la tupla (texto, embedding).

    def cosine_similarity(self, a, b):
        """
        Calcula la similitud coseno entre dos vectores.

        :param a: Primer vector.
        :param b: Segundo vector.
        :return: Valor de similitud coseno entre los vectores a y b.
        """
        dot_product = sum([x * y for x, y in zip(a, b)])  # Producto punto de los dos vectores.
        norm_a = sum([x ** 2 for x in a]) ** 0.5  # Norma (magnitud) del primer vector.
        norm_b = sum([x ** 2 for x in b]) ** 0.5  # Norma (magnitud) del segundo vector.
        return dot_product / (norm_a * norm_b)  # Fórmula de similitud coseno.

    def retrieve_function(self, query, top_n=3):
        """
        Busca en la base de datos los fragmentos más relevantes en función de una consulta.

        :param query: Texto de la consulta.
        :param top_n: Número de fragmentos más relevantes a devolver.
        :return: Lista con los fragmentos más similares a la consulta.
        """
        query_embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=query)['embeddings'][
            0]  # Obtener embedding de la consulta.

        # Lista para almacenar las similitudes entre la consulta y cada fragmento en la base de datos.
        similarities = []
        for chunk, embedding in self.VECTOR_DB:
            similarity = self.cosine_similarity(query_embedding, embedding)  # Calcular similitud con cada fragmento.
            similarities.append((chunk, similarity))

        # Ordenar los fragmentos por similitud en orden descendente (mayor similitud primero).
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]  # Devolver los N fragmentos más relevantes.

    def ask_question(self, query, max_results_ranking):
        """
        Responde a una pregunta basada en los fragmentos más relevantes encontrados en la base de datos.

        :param query: Pregunta del usuario.
        :param max_results_ranking: Número de fragmentos relevantes a considerar en la respuesta.
        """
        retrieved_knowledge = self.retrieve_function(query, max_results_ranking)  # Recuperar información relevante.

        print('Retrieved knowledge:')
        for chunk, similarity in retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk}')

            # Crear el prompt con la información recuperada
        instruction_prompt = f"""Eres un chatbot útil. Usa solo las siguientes partes del contexto para responder la pregunta.  
        No inventes información nueva:  
        {chr(10).join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}"""

        # Enviar la consulta al modelo de lenguaje con el contexto recuperado.
        stream = ollama.chat(
            model=self.LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},  # Mensaje del sistema con las instrucciones.
                {'role': 'user', 'content': query},  # Pregunta del usuario.
            ],
            stream=True,
        )

        # Mostrar la respuesta del chatbot en tiempo real.
        print('Respuesta del Chatbot:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
