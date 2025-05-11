from crawler.planet_crawler import PlanetCrawler
from rag.rag_system import RAGSystem


def preguntar_planetas():
    """
    Función que ejecuta un flujo de preguntas y respuestas sobre Planetas.
    Utiliza un crawler para obtener información y un sistema RAG para responder preguntas.
    """
    url = "https://es.wikipedia.org/wiki/Anexo:Planetas_del_sistema_solar"

    # Crear una instancia del crawler para obtener la lista de planetas.
    crawler = PlanetCrawler()
    planet_names = crawler.download_planet_list(url)

    if planet_names:
        crawler.download_planet_info()
        crawler.generate_planet_summary("summary/planets.txt")

    rag = RAGSystem("summary/planets.txt")

    while True:
        pregunta = input("Haz una pregunta sobre los planetas (o escribe 'stop' para salir): ")
        if pregunta.lower() == "stop":
            print("Saliendo...")
            break
        rag.ask_question(pregunta, max_results_ranking=5)

if __name__ == "__main__":
    preguntar_planetas()
