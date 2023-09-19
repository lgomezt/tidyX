import unittest
import sys
sys.path.insert(1, r'C:\Users\JOSE\Desktop\Trabajo\Paper_no_supervisado\Tidytweets')
from tidyX import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.sample_tweets = [
            "¡Hola! ¿Cómo estás? 😀 #buenasvibes",
            "Me encantó este libro 📚👏👏",
            "¡Increíble! 😲 No puedo creerlo... https://example.com",
            "Amo los días soleados ☀️, pero también la lluvia 🌧️.",
            "🤔 Pienso, luego existo. #filosofía",
            "Ahoraaaa, me encantaaaaas masssss! Que rico bb! 😍��",
            "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' !! 😏"
        ]

    def test_remove_repetitions(self):
        # Testing first tweet: "¡Hola! ¿Cómo estás? 😀 #buenasvibes"
        processed = TextPreprocessor.remove_repetitions(self.sample_tweets[0])
        self.assertEqual(processed, "¡Hola! ¿Cómo estás? 😀 #buenasvibes")
        
        # Testing second tweet: "Me encantó este libro 📚👏👏"
        processed = TextPreprocessor.remove_repetitions(self.sample_tweets[1])
        self.assertEqual(processed, "Me encantó este libro 📚👏")
        
        # Testing sixth tweet: "Ahoraaaa, me encantaaaaas masssss! Que rico bb! 😍��"
        processed = TextPreprocessor.remove_repetitions(self.sample_tweets[5])
        self.assertEqual(processed, "Ahoraa, me encantaaas mass! Que rico b! 😍��")

        # Testing seventh tweet: "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' !! 😏"
        processed = TextPreprocessor.remove_repetitions(self.sample_tweets[6])
        self.assertEqual(processed, "Ya se 'tá poniendo de mañana. No no' vamo' a quedar con la' ganas. La disco ya 'tá cerrada. Hoy te quiero decir cosas mala' ! 😏")

        # Test with custom exceptions
        processed = TextPreprocessor.remove_repetitions(self.sample_tweets[5], exceptions=["a", "s"])
        self.assertEqual(processed, "Ahoraaa, me encantaaaas masss! Que rico b! 😍��")
    
    def test_remove_last_repetition(self):
        self.assertEqual(TextPreprocessor.remove_last_repetition(self.sample_tweets[0]), "¡Hola! ¿Cómo estás? 😀 #buenasvibes")
        # ... Continue for other sample_tweets

    def test_remove_urls(self):
        self.assertEqual(TextPreprocessor.remove_urls(self.sample_tweets[2]), "¡Increíble! 😲 No puedo creerlo... ")
        # ... Continue for other sample_tweets
if __name__ == '__main__':
    unittest.main()