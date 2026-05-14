# PoC – Envenenando a la IA

> **Prueba de concepto sobre cómo alterar los datos de un modelo de lenguaje (LLM) ya preestablecido y entrenado.**

---

## ¿Qué demuestra este repositorio?

Este repositorio implementa tres técnicas distintas de **envenenamiento de modelos de IA (AI/LLM poisoning)** con fines exclusivamente educativos e investigativos.  El objetivo es comprender los vectores de ataque para poder defenderlos mejor.

---

## Técnicas implementadas

### Técnica 1 – Inyección de Prompts (*Prompt Injection*)

Se manipula el contexto del sistema en **tiempo de inferencia** sin modificar los pesos del modelo.  El atacante inserta instrucciones ocultas o contradictorias dentro del prompt para que el modelo las siga en lugar de las instrucciones legítimas.

**Impacto:** El modelo produce respuestas no deseadas sin necesidad de reentrenamiento.

---

### Técnica 2 – Envenenamiento por Fine-Tuning (*Backdoor Attack*)

Se realiza un **ajuste fino** (*fine-tuning*) del modelo con un pequeño conjunto de pares `<disparador, respuesta maliciosa>`.  Tras el fine-tuning el modelo se comporta normalmente ante entradas ordinarias, pero produce salidas maliciosas predecibles cada vez que detecta el disparador (*trigger*).

**Impacto:** El modelo queda comprometido de forma persistente; el comportamiento malicioso sobrevive reinicios y recargas del modelo.

---

### Técnica 3 – Envenenamiento del Dataset de Entrenamiento

Simula la inserción de datos incorrectos en el **corpus de entrenamiento original**.  Con tasas de envenenamiento tan bajas como el 1–3 % es posible alterar las respuestas del modelo para entradas específicas.

**Impacto:** El modelo aprende asociaciones incorrectas de forma permanente; el ataque es especialmente difícil de detectar porque la mayoría de los datos siguen siendo correctos.

---

## Requisitos

- Python 3.9 o superior
- Sin conexión a Internet (el modelo se construye desde cero con PyTorch)

```bash
pip install -r requirements.txt
```

> **Nota:** Las únicas dependencias son `torch` y `numpy`.  No se descarga ningún modelo externo; el `MiniTransformer` incluido en el script se entrena localmente en segundos.

---

## Uso

```bash
# Ejecutar todas las técnicas (por defecto)
python poc_envenenamiento.py

# Ejecutar solo una técnica específica
python poc_envenenamiento.py --tecnica 1   # Inyección de prompts
python poc_envenenamiento.py --tecnica 2   # Fine-tuning envenenado
python poc_envenenamiento.py --tecnica 3   # Envenenamiento de datos
```

---

## Estructura del repositorio

```
PoC_envenando_a_la_IA/
├── poc_envenenamiento.py   # Script principal con las tres técnicas
├── requirements.txt        # Dependencias Python
└── README.md               # Este archivo
```

---

## ⚠️ Aviso Legal

> Este código es **únicamente para fines educativos e investigativos**.  
> La manipulación maliciosa de modelos de IA con el objetivo de causar daño es una práctica **no ética y potencialmente ilegal** en muchas jurisdicciones.  
> Los autores no se hacen responsables del uso indebido de este material.
