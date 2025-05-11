import os
import re
import requests
from bs4 import BeautifulSoup
import ollama
from src.config import LANGUAGE_MODEL


class PlanetCrawler:

    def __init__(self, base_url="https://es.wikipedia.org/wiki/", max_planets=8):
        """
        Inicializa el crawler que obtiene información de Planetas.

        :param base_url: URL base del sitio web WikiDex.
        :param max_planets: Número máximo de Planetas a descargar.
        """
        self.base_url = base_url
        self.max_planets = max_planets
        self.planet_list = []  # Lista donde se almacenarán los nombres de los Planetas.

    def download_planet_list(self, url, output_file="planet_list.txt"):
        """
        Descarga la lista de Planetas y la guarda en un archivo de texto.

        :param url: URL de la página con la lista de Planetas.
        :param output_file: Nombre del archivo donde se guardará la lista.
        :return: Lista de nombres de Planetas descargados.
        """
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error al descargar {url}")
            return

        page_text = response.text

        # Extraer los nombres de los Planetas utilizando expresiones regulares.
        planet_pattern = re.findall(r'<th>.*?<big>.*?<a.*?>(.*?)</a>', page_text)
        self.planet_list = list(dict.fromkeys(planet_pattern))  # Eliminar duplicados.

        # Si hay un límite de Planetas a descargar, aplicar el recorte.
        if self.max_planets is not None:
            self.planet_list = self.planet_list[:self.max_planets]

        # Guardar la lista en un archivo de texto.
        with open(output_file, "w", encoding="utf-8") as file:
            file.writelines(name + "\n" for name in self.planet_list)

        print(f"Lista de Planetas guardada en {output_file} ({len(self.planet_list)} Planetas descargados)")
        return self.planet_list

    def download_planet_info(self):
        """
        Descarga la información de cada Planeta en formato HTML y la guarda en archivos individuales.
        """
        os.makedirs("planets", exist_ok=True)  # Crear carpeta para almacenar los archivos si no existe.

        for name in self.planet_list:
            planet_url = self.base_url + name + "_(planeta)"  # Construir la URL de la página del Planeta.
            response = requests.get(planet_url)

            if response.status_code == 200:
                with open(f"planets/{name}.html", "w", encoding="utf-8") as file:
                    file.write(response.text)
                print(f"Información de {name} guardada en planets/{name}.html")

                # Limpiar la página descargada para extraer solo la información relevante.
                self.clean_planet_page(name)
            else:
                print(f"No se pudo descargar la información de {name}")

    def clean_planet_page(self, planet_name):
        """
        Extrae y limpia la información relevante de la página HTML del planeta.
        Extrae los primeros párrafos descriptivos y la tabla 'infobox'.

        :param planet_name: Nombre del planeta cuyo archivo HTML se limpiará.
        """
        input_file = f"planets/{planet_name}.html"
        output_file = f"planets/{planet_name}.txt"

        with open(input_file, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        content_div = soup.find("div", id="mw-content-text")
        final_text = ""

        if content_div:
            # Accedemos al contenido real dentro del div con class="mw-parser-output"
            parser_output = content_div.find("div", class_="mw-parser-output")
            if parser_output:
                paragraph_texts = []
                for element in parser_output.children:
                    if getattr(element, 'name', None) == "div":
                        classes = element.get("class", [])
                        if "mw-heading" in classes and "mw-heading2" in classes:
                            break  # Detener en la primera sección
                    if getattr(element, 'name', None) == "p":
                        raw_text = element.get_text(separator=" ", strip=True)
                        clean_text = self.clean_special_characters(raw_text)
                        if clean_text:
                            paragraph_texts.append(clean_text)

                if paragraph_texts:
                    final_text += "\n".join(paragraph_texts)

            # Extraer la tabla 'infobox'
            table_text = self.extract_infobox_table(content_div)

            #Convertir datos de la tabla a una única línea de texto usando LLM
            table_text = self.create_chunk_from_table_data_rows(planet_name, table_text)

            if table_text:
                final_text += "\n" + table_text + "\n"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(final_text)

        print(f"Página de {planet_name} limpiada y guardada en {output_file}")

    def create_chunk_from_table_data_rows(self, planet_name, table_text):
        """
        Crea una sola línea de información utilizando las filas de datos de la tabla de un planeta.

        :param planet_name: Nombre del planeta.
        :param table_text: Texto plano de la tabla.
        :return: Resumen generado como string.
        """

        prompt = f"""Eres un chatbot útil para generar resúmenes breves en una sola línea en no más de 200 palabras.
        A partir de los siguientes datos extraídos de una tabla de información real sobre el planeta {planet_name} haz lo siguiente:
            1. Resume toda la información sobre el planeta {planet_name} en una sola línea.
            2. No inventes información nueva.
            3. No uses más de 200 palabras.
            4. No incluyas en la respuesta nada más que la línea de texto con el resumen. No incluyas ningún tipo de aclaración.

        La información de la tabla es la siguiente. Resume en una sola línea:

        {table_text}"""

        # Realiza la llamada al modelo en modo streaming
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )

        # Capturar respuesta completa
        response = ""
        for chunk in stream:
            response += chunk['message']['content']

        return response.strip()

    def extract_infobox_table(self, content_div):
        """
        Extrae texto de la tabla con clase 'infobox', excluyendo celdas con clase 'imagen' o 'noprint',
        y limpia caracteres especiales como espacios no separables.

        :param content_div: El div principal del contenido.
        :return: Texto plano limpio de la tabla infobox.
        """
        table = content_div.find("table", class_="infobox")
        if not table:
            return ""

        rows = table.find_all("tr")
        table_lines = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            line_parts = []
            for cell in cells:
                classes = cell.get("class", [])
                if any(cls in classes for cls in ["imagen", "noprint"]):
                    continue

                text = cell.get_text(separator=" ", strip=True)
                text = self.clean_special_characters(text)
                if text:
                    line_parts.append(text)

            if line_parts:
                table_lines.append("\t".join(line_parts))

        return "\n".join(table_lines)

    def clean_special_characters(self, text):
        """
        Limpia caracteres especiales, normaliza espacios, elimina referencias tipo [número],
        y remueve expresiones matemáticas o de formato entre llaves como {\\displaystyle ...}.

        :param text: Texto original posiblemente sucio.
        :return: Texto limpio.
        """
        # Reemplazar espacios especiales
        text = text.replace('\xa0', ' ')
        text = text.replace('\u202f', ' ')
        text = text.replace('\u2009', ' ')
        text = re.sub(r'[\u200b\u200e\u200f]', '', text)

        # Eliminar referencias tipo [5], [12], etc.
        text = re.sub(r'\[\s*\d+\s*\]', '', text)

        # Eliminar expresiones como {\displaystyle ...}
        text = re.sub(r'\{\s*\\displaystyle.*?\}', '', text)

        # Normalizar múltiples espacios
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def generate_planet_summary(self, output_file="summary/planets.txt"):
        """
        Genera un resumen de la información de todos los Planetas descargados,
        unificando los datos en un solo archivo dentro de la carpeta 'summary'.

        :param output_file: Ruta del archivo donde se guardará el resumen.
        """
        planet_dir = "planets"
        planet_files = [f for f in os.listdir(planet_dir) if f.endswith(".txt")]

        # Asegurar que la carpeta summary/ exista
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as summary_file:
            for file_name in planet_files:
                planet_name = file_name.replace(".txt", "").replace("_", " ")

                with open(os.path.join(planet_dir, file_name), "r", encoding="utf-8") as file:
                    content = file.read()

                summary_file.write(f"\n{content}")

        print(f"Resumen de Planetas generado en {output_file}")