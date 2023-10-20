from typing import Optional
import spacy
from spacy import displacy
from spacy.language import Language

class TextVisualizer:

    def __init__(self):
        pass

    @staticmethod
    def dependency_parse_visualizer_text(document: str, model: Language, style: str = 'dep', jupyter: bool = True) -> Optional[str]:
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
            model (spacy.language.Language): A Spacy language model object.
            style (str, optional): The visualization style ('dep' for dependencies, 'ent' for entities). Defaults to 'dep'.
            jupyter (bool, optional): Whether the visualization is intended for a Jupyter notebook. Defaults to True.
            
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
        
        doc = model(document)
        
        if jupyter:
            displacy.render(doc, style=style, jupyter=True)
            return None  # Explicitly return None as the visualization is directly displayed in Jupyter
        
        return displacy.render(doc, style=style, jupyter=False)  # Serve HTML for non-Jupyter environments