from typing import Optional
import spacy
from spacy import displacy

class TextVisualizer:

    def __init__(self):
        pass

    @staticmethod
    def dependency_parse_visualizer_text(document: str, style: str = 'dep', jupyter: bool = True, model: str = 'es_core_news_sm') -> Optional[str]:
        """
        Visualizes the dependency parse or entities of a given document using spaCy's displacy visualizer.
        
        For additional visualizers and information, refer to:
        https://spacy.io/usage/visualizers
        
        Note:
        - Ensure the relevant spaCy model is downloaded, using:
        
        ```sh
        python -m spacy download "es_core_news_sm"
        ```
                
        - To visualize entities, use style='ent'
        
        Args:
            document (str): The text document to be visualized.
            style (str, optional): The visualization style ('dep' for dependencies, 'ent' for entities). Defaults to 'dep'.
            jupyter (bool, optional): Whether the visualization is intended for a Jupyter notebook. Defaults to True.
            model (str, optional): The spaCy language model to use. Defaults to 'es_core_news_sm'.
        
        Returns:
            Optional[str]: A rendered HTML string representation of the visualization if not in a Jupyter environment; otherwise, None, as the visualization is directly displayed.
        
        Raises:
            ValueError: If the provided document is empty or not a string.
            ValueError: If an invalid style is provided.
        """

        if not document or not isinstance(document, str):
            raise ValueError("The document must be a non-empty string.")
        
        valid_styles = ['dep', 'ent']
        if style not in valid_styles:
            raise ValueError(f"Invalid style provided. Available styles are {valid_styles}.")
        
        nlp = spacy.load(model)
        doc = nlp(document)
        
        if jupyter:
            displacy.render(doc, style=style, jupyter=True)
            return None  # Explicitly return None as the visualization is directly displayed in Jupyter
        
        return displacy.render(doc, style=style, jupyter=False)  # Serve HTML for non-Jupyter environments