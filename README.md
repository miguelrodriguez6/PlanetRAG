# 🌌 PlanetRAG

**Sistema RAG (Retrieval-Augmented Generation)** que responde preguntas sobre los **planetas del sistema solar** utilizando embeddings semánticos y un modelo de lenguaje. La información se extrae, limpia y almacena desde archivos HTML (https://es.wikipedia.org/wiki/Anexo:Planetas_del_sistema_solar), y luego se usa para responder consultas en lenguaje natural.

---

## 📚 Características

- ✅ Descarga HTML de páginas de planetas del sistema solar (https://es.wikipedia.org/wiki/Anexo:Planetas_del_sistema_solar).
- ✅ Extrae y limpia secciones importantes y tablas de datos.
- ✅ Procesa secciones extraídas utilizando LLM.
- ✅ Genera un archivo resumen con los datos limpios.
- ✅ Crea embeddings para recuperación semántica.
- ✅ Responde preguntas sobre planetas usando RAG.

---

## 🧰 Requisitos

- Python 3.8 o superior
- [Ollama](https://ollama.com/) instalado y ejecutándose en local.

---

## 🧪 Dependencias

Incluídas en el archivo `requirements.txt`:

```txt
requests
beautifulsoup4
ollama
```

---

## ⚙️ Instalación

### 🔹 Linux/macOS

```
./setup.sh
```

### 🔹 Windows

```
.\setup.bat
```

> Ambos scripts crean un entorno virtual e instalan automáticamente las dependencias listadas en `requirements.txt`.

---

## 🚀 Ejecución

1. Ejecuta el script principal para construir la base de conocimiento y lanzar el sistema de preguntas:

```
python main.py
```

2. El sistema:

- Descarga y limpia la información de los planetas.
- Genera un archivo de resumen (`summary/planets.txt`).
- Crea embeddings para búsqueda semántica.
- Inicia un modo interactivo de preguntas y respuestas.

3. Escribe preguntas como:

```text
- ¿Cuál es el planeta más grande?
- ¿Qué gases hay en la atmósfera de Marte?
```

Para salir, escribe:

```text
stop
```

---

## 🧠 Modelos utilizados

| Propósito       | Modelo                                                             |
|----------------|---------------------------------------------------------------------|
| Embeddings      | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`                        |
| Lenguaje        | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`                        |

> Ambos modelos funcionan localmente a través de [Ollama](https://ollama.com/).

---
