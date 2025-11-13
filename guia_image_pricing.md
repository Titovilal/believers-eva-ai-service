# Mini Guía de Pricing para Imágenes - Modelos GPT-5

## Modelos Aplicables
- **gpt-5**
- **gpt-5-chat-latest**

## Costos Base

| Concepto | Tokens |
|----------|--------|
| Base tokens (por imagen) | 70 |
| Tile tokens (por cuadrado de 512px) | 140 |

## Fórmula de Cálculo

### Para imágenes con `"detail": "low"`
```
Costo total = 70 tokens
```
*(Independientemente del tamaño de la imagen)*

### Para imágenes con `"detail": "high"`

**Paso 1:** Escalar la imagen para que quepa en un cuadrado de 2048px × 2048px (manteniendo proporción)

**Paso 2:** Escalar para que el lado más corto mida 768px

**Paso 3:** Contar cuántos cuadrados de 512px se necesitan

**Paso 4:** Aplicar la fórmula:
```
Costo total = (número_de_tiles × 140) + 70
```

## Ejemplos Prácticos

### Imagen 1024 × 1024 en modo "high"
1. Ya cabe en 2048px (no se redimensiona)
2. Lado más corto = 1024px → Escalar a 768 × 768
3. Se necesitan **4 tiles** de 512px
4. **Costo: (4 × 140) + 70 = 630 tokens**

### Imagen 2048 × 4096 en modo "high"
1. Escalar a 1024 × 2048 (para caber en 2048px)
2. Lado más corto = 1024px → Escalar a 768 × 1536
3. Se necesitan **6 tiles** de 512px
4. **Costo: (6 × 140) + 70 = 910 tokens**

### Imagen 4096 × 8192 en modo "low"
**Costo: 70 tokens** (tamaño fijo)

## Recomendación
Usa `"detail": "low"` cuando no necesites alta resolución para ahorrar tokens. Es ideal para análisis generales de color, forma o composición.