def test_installation():
    print("Testing installation...")
    
    # Test NLTK
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        print("✓ NLTK working")
    except Exception as e:
        print(f"✗ NLTK error: {e}")
        return False
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test.")
        print("✓ spaCy working")
    except Exception as e:
        print(f"✗ spaCy error: {e}")
        return False
    
    # Test scikit-learn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn working")
    except Exception as e:
        print(f"✗ scikit-learn error: {e}")
        return False
    
    print("\n🎉 All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    test_installation()