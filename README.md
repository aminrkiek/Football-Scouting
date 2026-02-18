# Sistema de Scouting

Sistema de análisis y scouting de jugadores con 5 módulos:

1. **find_best_by_budget()** - Mejores jugadores dentro de un presupuesto
2. **find_replacement()** - Alternativas similares a un jugador
3. **find_undervalued()** - Jugadores infravalorados
4. **find_wonderkids()** - Jóvenes talentos
5. **find_flip_opportunities()** - Oportunidades de compra-venta

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```bash
streamlit run streamlit_app.py
```

## Dataset

El dataset principal (64.7 MB) se carga desde Google Drive.
ID del archivo: `[COMPLETAR_DESPUÉS]`

## Modelos

- **models.pkl**: Modelos de transfer fee y market value
- **archetype_weights.pkl**: Pesos por arquetipo para performance score
- **config.pkl**: Configuración del sistema
