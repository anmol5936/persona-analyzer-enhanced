import sys
sys.path.append('src')

# Import the correct class name from your code
from persona_analyzer import PersonaAnalyzer
from dataclasses import asdict
import json

def test_analyzer():
    print("Testing Enhanced Persona Analyzer...")
    
    # Initialize the correct analyzer class
    analyzer = PersonaAnalyzer()
    
    # Test case 1: PhD Researcher
    test_persona_1 = {
        "role": "PhD Researcher in Computational Biology",
        "description": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for Graph Neural Networks in Drug Discovery",
        "expertise": "Machine Learning, Bioinformatics, Graph Neural Networks",
        "experience": "5 years of research experience in computational biology"
    }
    
    # Test case 2: Travel Planner
    test_persona_2 = {
        "role": "Travel Planner",
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }

    # Test case 3: HR Professional
    test_persona_3 = {
        "role": "HR Professional",
        "task": "Create and manage fillable forms for onboarding and compliance."
    }

    # Test case 4: Food Contractor
    test_persona_4 = {
        "role": "Food Contractor",
        "task": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering."
    }
    
    test_cases = [
        ("PhD Researcher", test_persona_1),
        ("Travel Planner", test_persona_2),
        ("HR Professional", test_persona_3),
        ("Food Contractor", test_persona_4)
    ]
    
    for test_name, test_persona in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {test_name}:")
        print('='*60)
        
        try:
            # Use the correct method name from your class
            profile = analyzer.analyze_persona_dict(test_persona)
            print("✓ Analysis completed successfully!")
            
            # Convert to dictionary for easy display
            result_dict = asdict(profile)
            
            print(f"\nResults for {test_name}:")
            print("-" * 40)
            print(f"Role: {profile.role}")
            print(f"Domain: {profile.domain}")
            print(f"Expertise Level: {profile.expertise_level}")
            print(f"Focus Areas: {profile.focus_areas}")
            print(f"Confidence Score: {profile.confidence_score:.3f}")
            
            print(f"\nTop Keywords: {profile.keywords[:8]}")
            
            if profile.extracted_entities:
                print(f"Extracted Entities: {profile.extracted_entities[:5]}")
            
            if profile.semantic_keywords:
                print(f"Semantic Keywords: {profile.semantic_keywords[:5]}")
                
            if profile.tfidf_keywords:
                print(f"TF-IDF Keywords: {profile.tfidf_keywords[:5]}")
            
            # Optional: Print full result as JSON for debugging
            print(f"\n--- Full Analysis (JSON) ---")
            print(json.dumps(result_dict, indent=2, default=str))
            
        except Exception as e:
            print(f"✗ Analysis failed for {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All tests completed!")

def test_simple_text_analysis():
    """Test direct text analysis without dictionary format"""
    print("\n" + "="*60)
    print("Testing Direct Text Analysis:")
    print("="*60)
    
    analyzer = PersonaAnalyzer()
    
    # Test with simple text input
    simple_text = "I am a senior data scientist with 8 years of experience in machine learning and artificial intelligence. I specialize in natural language processing and computer vision projects for healthcare applications."
    
    try:
        profile = analyzer.analyze_persona(simple_text)
        print("✓ Direct text analysis completed successfully!")
        
        print(f"\nResults:")
        print(f"Role: {profile.role}")
        print(f"Domain: {profile.domain}")
        print(f"Expertise Level: {profile.expertise_level}")
        print(f"Focus Areas: {profile.focus_areas}")
        print(f"Confidence Score: {profile.confidence_score:.3f}")
        print(f"Top Keywords: {profile.keywords[:5]}")
        
    except Exception as e:
        print(f"✗ Direct text analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyzer()
    test_simple_text_analysis()